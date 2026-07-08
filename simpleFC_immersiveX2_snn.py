import os
os.environ["XDG_SESSION_TYPE"] = "x11"
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import GC as g
import open3d as o3d
import math
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager as font_manager
import time
import csv
import os
import sys
import time
import subprocess
import threading

import torch
import torch.nn as nn
from loss import dice_loss, iou_score
from spikingjelly.activation_based import functional, monitor, neuron

#global voxel creation
#redwood
upper_bound = 3.0
lower_bound = -1.8
gridStep = 0.02

#own dataset
# upper_bound = 3.9
# lower_bound = -2.8
# gridStep = 0.02

def evaluate_snn(model, dataloader, config, log_interval=5):
    """
    Evaluate SNN model performance

    Args:
        snn_model: SNN model
        dataloader: Data loader
        config: Config object
        log_interval: Number of batches between logging

    Returns:
        Average loss and average IoU
    """
    device = config.device
    dice_weight = config.dice_weight
    bce_weight = config.bce_weight
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    transmission_time, client_time, server_time = 0.0, 0.0 , 0.0
    bce = nn.BCEWithLogitsLoss()
    T = config.snn_steps
    batch_count = 0

    Smonitor = monitor.OutputMonitor(model, neuron.IFNode)
    for batch_idx, (pts, vox_gt) in enumerate(dataloader, 1):
        start_client = time.time()
        pts = pts.to(device, non_blocking=True)  # [B, N, 3]
        vox_gt = vox_gt.to(device, non_blocking=True)  # [B, G, G, G]

        # Reset network state
        functional.reset_net(model)


        # Run through multiple timesteps and accumulate outputs
        y_sum = 0
        total_spikes =0
        total_neu =0
        for t in range(T):
            y_t = model(pts) # [B, G, G, G]
            y_sum += y_t

        # Use average output as final prediction
        y_mean = y_sum / T

        end_client = time.time()

        # Calculate loss and IoU
        for rec in Smonitor.records:
            total_spikes += (rec > 0).sum().item()
            total_neu += rec.numel()

        Smonitor.clear_recorded_data()

        spike_lambda = 1e-3
        spike_rate = total_spikes / (total_neu + 1e-9)
        loss = bce_weight * bce(y_mean, vox_gt) + (dice_weight * dice_loss(y_mean, vox_gt))
        # loss = loss + spike_lambda * np.log(1 + spike_rate)
        loss = loss + spike_lambda * (spike_rate ** 2)
        total_loss += loss.item()
        total_iou += iou_score(y_mean, vox_gt).item()
        batch_count += 1

        total_packet = 4096 / (1460 * 8)
        temp_time = (total_packet * 0.001) + 0.02
        temp_time = temp_time * T
        transmission_time += temp_time

        # Log batch progress
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / batch_count
            avg_iou = total_iou / batch_count
            spike_rate = total_spikes / (total_neu + 1e-9)
            print(f"Val Batch {batch_idx:03d} | Loss {avg_loss:.4f} | IoU {avg_iou:.3f} | Spike rate {spike_rate:.6f}")


        start_server = time.time()
        probs = torch.sigmoid(y_mean).detach().cpu().numpy()[0]  # [G,G,G]
        pred_bin = (probs >= 0.5).astype(np.float32)
        pred_vox = recons_voxel(pred_bin, offset=np.array([-1.2, 0.0, 0.0]))

        end_server = time.time()

        client_time += end_client - start_client
        server_time += end_server - start_server

        # Convert voxels to colored point clouds (centers)
        vox = vox_gt.detach().cpu().numpy()[0]
        gt_vox = recons_voxel(vox, offset=np.array([1.2, 0.0, 0.0]))

        enetime.append(end_client - start_client + temp_time + end_server - start_server)
        comtime.append(end_client - start_client)
        decomtime.append(end_server - start_server)
        transtime.append(temp_time)


        # o3d.visualization.draw_geometries([gt_vox])
        # o3d.visualization.draw_geometries([pred_vox])

    final_spike_rate = total_spikes / (total_neu + 1e-9)
    print(f"Final Spike Rate {final_spike_rate:.6f}")
    print(f"Total Spikes {total_spikes:.6f}")
    n = len(dataloader)
    return total_loss / n, total_iou / n, client_time / n, transmission_time / n, server_time /n


def recons_voxel(voxel_grid_np, color=[1.0, 0.5, 0.0], voxel_size=0.04, offset=np.array([0.0, 0.0, 0.0])):
    """
    Convert a binary voxel numpy array [G,G,G] into a colored point cloud
    at the centers of occupied voxels (for Open3D visualization).
    """
    occ = np.argwhere(voxel_grid_np > 0.5)  # (K,3), indices in [0..G-1]
    if occ.shape[0] == 0:
        # return empty pcd to avoid Open3D errors
        return o3d.geometry.PointCloud()
    centers = (occ.astype(np.float32) + 0.5) * voxel_size + offset
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    # pcd.paint_uniform_color(color)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

def get_time(data):
    bit = np.unpackbits(np.array(data, dtype=np.uint8), axis=1, count=64)
    bit = bit.flatten()
    total_packet = len(bit) / (1460 * 8)
    trans_time = (total_packet * 0.001) + 0.02
    return trans_time

def cal_firing_rate(s_seq: torch.Tensor):
    return  s_seq.flatten(1).mean(1)


def power_logger(log_path, signal_ref, stop_event):
    with open(log_path, "w") as f:
        f.write("timestamp,power_W,signal\n")
        while not stop_event.is_set():
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=timestamp,power.draw",
                "--format=csv,noheader" ], capture_output=True, text=True)
            line = result.stdout.strip()
            if line:
                ts, pwr = line.split(",")
                pwr = pwr.strip().replace(" W", "")
                f.write(f"{ts},{pwr},{signal_ref[0]}\n")
                f.flush()
                time.sleep(0.05)


# def load_fine_tuned_snn(encoder_checkpoint_path,decoder_checkpoint_path, device, mode):
#     """
#     Load fine-tuned SNN model
#
#     Args:
#         checkpoint_path: Model checkpoint path
#         device: Device
#
#     Returns:
#         SNN model
#     """
#     from model import AE
#     from dataset import PcdDataset
#     from torch.utils.data import DataLoader
#
#     config = Config()
#
#     # Load original ANN model as reference for conversion
#     print("Loading ANN model...")
#     encoder_model = PointNetEncoder().to(device)
#     encoder_model.load_state_dict(torch.load("best_encoder_model_new.pth", map_location=device))
#
#     decoder_model = Conv3DDecoder().to(device)
#     decoder_model.load_state_dict(torch.load("best_decoder_model_new.pth", map_location=device))
#
#     # Prepare conversion dataset
#     tr_dataset = PcdDataset(config=config, split="train")
#     tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size_ann2snn, shuffle=True,
#                                collate_fn=collate_points)
#
#     # Convert to SNN
#     encoder_snn_model = convert_ann_to_snn(encoder_model, tr_dataloader, device, mode=mode)
#     encoder_snn_model = encoder_snn_model.to(device)
#
#     decoder_snn_model = convert_ann_to_snn(decoder_model, tr_dataloader, device, mode=mode)
#     decoder_snn_model = decoder_snn_model.to(device)
#
#     # Load fine-tuned weights
#     encoder_snn_model.load_state_dict(torch.load(encoder_checkpoint_path, map_location=device))
#     decoder_snn_model.load_state_dict(torch.load(decoder_checkpoint_path, map_location=device))
#     print(f"Loaded fine-tuned SNN model: {encoder_checkpoint_path}, {decoder_checkpoint_path}")
#
#     return encoder_snn_model, decoder_snn_model


# total_time=[]
# avg_time = []
# avg_compression_rate = []
# avg_Bcompression_rate = []
# avg_comp_time = []
# avg_trans_time = []

compression_rate = []
Bcompression_rate = []
enetime = []
comtime = []
decomtime = []
transtime = []
rmse =[]
avg_num=0


if __name__ == "__main__":
    from config import Config
    from model import PointNetEncoder, Conv3DDecoder, AE
    from dataset import PcdDataset
    from torch.utils.data import DataLoader
    from ann2snn import convert_ann_to_snn, collate_points, load_fine_tuned_snn
    from utils import estimate_pos_weight

    config = Config()
    device = config.device

    val_dataset = PcdDataset(config=config, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_ann2snn_test,
        shuffle=False,
        collate_fn=collate_points,
        num_workers=config.num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )

    bast_val_loss = float('inf')


    modes = ['max', '99.9%', 1/2, 1/3, 1/4, 1/5]
    for mode in modes:

        snn_model = load_fine_tuned_snn("MFT_best_snn_model_" + str(mode) + ".pth", device, mode)
        snn_model = snn_model.to(device)


        # signal_ref = [0]
        # stop_event = threading.Event()
        # power_thread = threading.Thread(target=power_logger, args=("snn_gpu_power_" + str(mode) + ".csv", signal_ref, stop_event))
        # power_thread.start()

        # time.sleep(5)  # idle baseline — signal=0

        # signal_ref[0] = 1  # inference starting
        # print("Inference started...")

        for epoch in range(1, config.epochs + 1):
            test_loss, test_iou, clientTime, TransmissionTime, serverTime = evaluate_snn(snn_model,  val_dataloader, config, log_interval=5)
            print(serverTime,TransmissionTime,clientTime)
            print(f"Epoch {epoch:03d}/{config.epochs} | test Loss {test_loss:.4f} IoU {test_iou:.3f} e2e time {clientTime+TransmissionTime+serverTime:.3f} encoding/decoding time {clientTime+serverTime:.3f} trans time {TransmissionTime:.3f} ")



        # signal_ref[0] = 0  # inference ended
        # time.sleep(5)  # post-inference idle — signal=0
        # stop_event.set()
        # power_thread.join()
        # print("Inference ended.")

        with open("SC_time_SNN_4voxel9" + str(mode) + ".csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(enetime)

        with open("SC_comtime_SNN_4voxel9" + str(mode) + ".csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(comtime)

        with open("SC_decomtime_SNN_4voxel9" + str(mode) + ".csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(decomtime)

        with open("SC_transtime_SNN_4voxel9" + str(mode) + ".csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(transtime)

'''
#packet
rawbit = np.unpackbits(np.array(queries, dtype=np.uint8), axis=1,count=64*3)
rawbit = rawbit.flatten()
print(len(rawbit),"raw")



bit = np.unpackbits(np.array(colors, dtype=np.uint8), axis=1,count=64)
print(bit, "SC")
bit = bit.flatten()
print(len(bit),"SC")
total_packet = len(bit)/(1460*8)
#ubit = np.packbits(bit, axis=-1,bitorder='big').view(np.uint32)

#entropy = g.entropy_count(len(Clookup)+1,colors)
#print(len(colors), entropy)
#min_col = set(colors)
#print("pcd",i,len(min_col))

comp_data_size = len(bit)
#print(comp_data_size)
comp_ratio = 100-(comp_data_size)*100/len(rawbit)
compression_rate.append(round(comp_ratio,2))

Bcomp_ratio = 100 - (len(csbit)) * 100 / len(rawbit)
Bcompression_rate.append(round(Bcomp_ratio,2))
#print("Compression rate: ",comp_ratio)

#destination
send_voxel=[]
start_decod_lookup = time.time()
for vc in colors:
    send_voxel.append(Clookup[vc[0]])
send_voxel = np.array(send_voxel)
re_voxel = reconstruct(send_voxel,Gvoxel,i,pc)
end_decod_lookup = time.time()

val = val.tolist()
#val.sort()
re_voxel = re_voxel.tolist()
#re_voxel.sort()
print(len(val), len(re_voxel))
t = []
if len(val)<len(re_voxel):
    for r in range(len(val)):
        t.append(np.square(np.subtract(val[r], re_voxel[r])))

else:
    for r in range(len(re_voxel)):
        t.append(np.square(np.subtract(val[r], re_voxel[r])))
t = np.array(t)
rmse.append(math.sqrt(np.mean(t)))

ttime = end_lookup-start_lookup+(total_packet*0.001)+0.001+end_decod_lookup-start_decod_lookup
print("end to end time for SC: ",i,"compression time (offline): ", end_compress-start_compress,"s online: ", ttime,math.sqrt(np.mean(t)))
avg_num = avg_num+len(re_voxel)
comtime.append(end_lookup-start_lookup)
transtime.append((total_packet*0.001)+0.001)
decomtime.append(end_decod_lookup-start_decod_lookup)
enetime.append(ttime)

# avg_compression_rate.append(compression_rate)
# avg_Bcompression_rate.append(Bcompression_rate)
# #avg_time.append(enetime)
# avg_comp_time.append(comtime)
# avg_trans_time.append(transtime)
#total_time.append(sum(enetime))
print("Sum for 40 pcd: ", sum(enetime))

print(avg_num/56)'''


'''# with open('rmse_4voxelvs3.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(rmse)


avg =[]
# for i in range(14):
#     avg.append(mean([avg_time[0][i],avg_time[1][i],avg_time[2][i],avg_time[3][i]]))
with open('SCtimeall.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(4):
        writer.writerow(avg_time[i])'''

# avg_Bcomp =[]
# avg_comp =[]
# avg_c = []
# avg_t = []
# for i in range(14):
#     # avg_c.append(mean([avg_comp_time[0][i], avg_comp_time[1][i], avg_comp_time[2][i], avg_comp_time[3][i],avg_comp_time[4][i], avg_comp_time[5][i], avg_comp_time[6][i], avg_comp_time[7][i],avg_comp_time[8][i], avg_comp_time[9][i], avg_comp_time[10][i], avg_comp_time[11][i]]))
#     # avg_t.append(mean([avg_trans_time[0][i], avg_trans_time[1][i], avg_trans_time[2][i], avg_trans_time[3][i],avg_trans_time[4][i], avg_trans_time[5][i], avg_trans_time[6][i], avg_trans_time[7][i],avg_trans_time[8][i], avg_trans_time[9][i], avg_trans_time[10][i], avg_trans_time[11][i]]))
#     # avg_comp.append(mean([avg_compression_rate[0][i],avg_compression_rate[1][i],avg_compression_rate[2][i],avg_compression_rate[3][i],avg_compression_rate[4][i],avg_compression_rate[5][i],avg_compression_rate[6][i],avg_compression_rate[7][i],avg_compression_rate[8][i],avg_compression_rate[9][i],avg_compression_rate[10][i],avg_compression_rate[11][i]]))
#     # avg_Bcomp.append(mean([avg_Bcompression_rate[0][i], avg_Bcompression_rate[1][i], avg_Bcompression_rate[2][i],avg_Bcompression_rate[3][i],avg_Bcompression_rate[4][i], avg_Bcompression_rate[5][i], avg_Bcompression_rate[6][i],avg_Bcompression_rate[7][i],avg_Bcompression_rate[8][i], avg_Bcompression_rate[9][i], avg_Bcompression_rate[10][i],avg_Bcompression_rate[11][i]]))
#
#     avg_c.append(mean([avg_comp_time[0][i], avg_comp_time[1][i], avg_comp_time[2][i], avg_comp_time[3][i]]))
#     avg_t.append(mean([avg_trans_time[0][i], avg_trans_time[1][i], avg_trans_time[2][i], avg_trans_time[3][i]]))
#     avg_comp.append(mean([avg_compression_rate[0][i],avg_compression_rate[1][i],avg_compression_rate[2][i],avg_compression_rate[3][i]]))
#     avg_Bcomp.append(mean([avg_Bcompression_rate[0][i], avg_Bcompression_rate[1][i], avg_Bcompression_rate[2][i],avg_Bcompression_rate[3][i]]))


# with open('avg_comp1D_4voxel.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(compression_rate)
#
# with open('avg_Bcomp1D_4voxel.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(Bcompression_rate)
#
#
# with open('avg_c1d_4voxel.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(comtime)
#
# with open('avg_d1d_4voxel.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(decomtime)
#
# with open('avg_t1d_4voxel.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(transtime)
