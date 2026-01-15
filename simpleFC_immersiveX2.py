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

import torch
import torch.nn as nn
from loss import dice_loss, iou_score

#global voxel creation
#redwood
upper_bound = 3.0
lower_bound = -1.8
gridStep = 0.02

#own dataset
# upper_bound = 3.9
# lower_bound = -2.8
# gridStep = 0.02

def eval_epoch(encoder, decoder, dataloader, config):
    device = config.device
    bce_pos_weight = config.bce_pos_weight
    dice_weight = config.dice_weight
    bce_weight = config.bce_weight
    encoder.eval()
    decoder.eval()
    total_loss, total_iou = 0.0, 0.0
    transmission_time, client_time, server_time = 0.0, 0.0 , 0.0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight], device=device))

    with (torch.no_grad()):
        for batch in dataloader:
            start_client = time.time()
            pts = batch["points"].to(device, non_blocking=True)  # [B, N, 3]
            vox_gt = batch["vox"].to(device, non_blocking=True)  # [B, G, G, G]

            vector_code = encoder(pts)
            vox_pred = decoder(vector_code)
            vox_pred = vox_pred.squeeze(1)
            end_client = time.time()

            loss = bce_weight * bce(vox_pred, vox_gt) + (dice_weight * dice_loss(vox_pred, vox_gt))

            total_loss += loss.item()
            total_iou += iou_score(vox_pred, vox_gt).item()


            vc = vector_code.detach().cpu().numpy()
            transmission_time +=get_time(vc)

            start_server = time.time()
            probs = torch.sigmoid(vox_pred).detach().cpu().numpy()[0]  # [G,G,G]
            pred_bin = (probs >= 0.5).astype(np.float32)
            pred_vox = recons_voxel(pred_bin, offset=np.array([-1.2, 0.0, 0.0]))

            end_server = time.time()

            client_time += end_client-start_client
            server_time += end_server - start_server

            # Convert voxels to colored point clouds (centers)
            vox = vox_gt.detach().cpu().numpy()[0]
            gt_vox = recons_voxel(vox, offset=np.array([1.2, 0.0, 0.0]))

            # o3d.visualization.draw_geometries([or_pts])
            #o3d.visualization.draw_geometries([gt_vox])
            #o3d.visualization.draw_geometries([pred_vox])



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
    from utils import estimate_pos_weight

    config = Config()
    encoder = PointNetEncoder().to(config.device)
    encoder_pth = torch.load("best_encoder_model.pth", map_location=config.device)
    encoder.load_state_dict(encoder_pth)

    decoder = Conv3DDecoder().to(config.device)
    decoder_pth = torch.load("best_decoder_model.pth", map_location=config.device)
    decoder.load_state_dict(decoder_pth)

    test_dataset = PcdDataset(config=config, split="val")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size_val_test,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
        persistent_workers=True
    )

    bast_val_loss = float('inf')
    for epoch in range(1, config.epochs + 1):
        test_loss, test_iou, clientTime, TransmissionTime, serverTime = eval_epoch(encoder, decoder, test_dataloader, config)
        print(serverTime,TransmissionTime,clientTime)
        print(f"Epoch {epoch:03d}/{config.epochs} | test Loss {test_loss:.4f} IoU {test_iou:.3f} e2e time {clientTime+TransmissionTime+serverTime:.3f} ")


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

print(avg_num/56)
# with open('SC_time40_12_4voxel.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(enetime)

# with open('rmse_4voxelvs3.csv', 'w', newline='') as file:
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
