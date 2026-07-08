import os
os.environ["XDG_SESSION_TYPE"] = "x11"
from spikingjelly.activation_based import ann2snn, functional, monitor, neuron
import torch
from loss import iou_score
import numpy as np
import open3d as o3d
from config import Config


def collate_points(batch):
    """
    Collate function for PcdDataset that converts dictionary format to (points, vox) format
    expected by SpikingJelly converter.
    
    Args:
        batch: list of dictionaries with keys "points", "colors", "vox"
               where "points" is [N, 3] numpy array or tensor
               and "vox" is [G, G, G] numpy array or tensor
    
    Returns:
        points: [B, N, 3] tensor - batched point clouds
        vox: [B, G, G, G] tensor - batched voxel grids
    """
    points_list = []
    vox_list = []
    
    for item in batch:
        if isinstance(item, dict):
            # Extract points and vox from dictionary
            points = item["points"]
            vox = item["vox"]
            
            # Convert points to tensor if needed
            if isinstance(points, torch.Tensor):
                points_list.append(points.contiguous())
            else:
                points_list.append(torch.from_numpy(points).float().contiguous())
            
            # Convert vox to tensor if needed
            if isinstance(vox, torch.Tensor):
                vox_list.append(vox.contiguous())
            else:
                vox_list.append(torch.from_numpy(vox).float().contiguous())
        else:
            # If already in tuple format (x, y), use directly
            points = item[0] if isinstance(item, (tuple, list)) else item
            vox = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else None
            
            if isinstance(points, torch.Tensor):
                points_list.append(points.contiguous())
            else:
                points_list.append(torch.from_numpy(points).float().contiguous())
            
            if vox is not None:
                if isinstance(vox, torch.Tensor):
                    vox_list.append(vox.contiguous())
                else:
                    vox_list.append(torch.from_numpy(vox).float().contiguous())
            else:
                # If no vox provided, create dummy vox (shouldn't happen with PcdDataset)
                raise ValueError("Vox data not found in batch item")
    
    # Stack into batch
    points_batch = torch.stack(points_list, dim=0)  # [B, N, 3]
    vox_batch = torch.stack(vox_list, dim=0)  # [B, G, G, G]
    
    return points_batch, vox_batch

def convert_ann_to_snn(model, dataloader, device, mode):
    print(f"Converting ANN to SNN (mode={mode})...")
    converter = ann2snn.Converter(dataloader=dataloader, device=str(device), mode=mode)
    snn_model = converter(model)
    return snn_model

def load_fine_tuned_snn(checkpoint_path, device,mode):
    """
    Load fine-tuned SNN model
    
    Args:
        checkpoint_path: Model checkpoint path
        device: Device
        
    Returns:
        SNN model
    """
    from model import AE
    from dataset import PcdDataset
    from torch.utils.data import DataLoader
    
    config = Config()
    
    # Load original ANN model as reference for conversion
    ann_model = AE().to(device)
    ann_model.load_state_dict(torch.load("best_model_new.pth", map_location=device))
    
    # Prepare conversion dataset
    tr_dataset = PcdDataset(config=config, split="train")
    tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size_ann2snn, shuffle=True, collate_fn=collate_points)
    
    # Convert to SNN
    snn_model = convert_ann_to_snn(ann_model, tr_dataloader, device, mode=mode)
    snn_model = snn_model.to(device)
    
    # Load fine-tuned weights
    snn_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded fine-tuned SNN model: {checkpoint_path}")

    return snn_model

@torch.no_grad()
def evaluate_snn_over_time(Smonitor,snn_model, test_loader, device, mode, T):
    snn_model.eval()
    print(f"Evaluating SNN for {T} timesteps (mode={mode})...")

    total_iou_t = np.zeros(T, dtype=np.float32)
    total_spike_rate_t = np.zeros(T, dtype=np.float32)
    n_batches = 0
    for xb_cpu, yb_cpu in test_loader:
        xb = xb_cpu.to(device, non_blocking=True)  # [B, N, 3]
        yb = yb_cpu.to(device, non_blocking=True)  # [B, G, G, G]
        total_spikes = 0
        total_neu = 0
        functional.reset_net(snn_model)
        y_sum = 0
        for t in range(T):
            y_t = snn_model(xb)
            y_sum += y_t
            y_mean = y_sum / (t + 1)  # running mean over timesteps
            for rec in Smonitor.records:
                total_spikes += (rec > 0).sum().item()
                total_neu += rec.numel()

            Smonitor.clear_recorded_data()

            total_spike_rate_t [t] += total_spikes*t / (total_neu + 1e-9)

            # # visualize voxels for the last timestep
            # if t == T - 1:
            #     from utils import visualize_voxels_to_file, recover_depth_points_from_voxels
            #     vox_pred_np = y_t.detach().cpu().numpy()[0]
            #     vox_gt_np = yb.detach().cpu().numpy()[0]
            #     grid_size = vox_pred_np.shape[0]
            #     pred_points = recover_depth_points_from_voxels(vox_pred_np, grid_size)
            #     gt_points = recover_depth_points_from_voxels(vox_gt_np, grid_size)
            #
            #     pcd_pred = o3d.geometry.PointCloud()
            #     pcd_pred.points = o3d.utility.Vector3dVector(pred_points)
            #     pred_colors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(pred_points), 1))
            #     pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
            #     pred_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_pred, voxel_size=0.04)
            #     # o3d.io.write_point_cloud(f"points_pred_snn_{mode}.ply", pcd_pred)
            #
            #     pcd_gt = o3d.geometry.PointCloud()
            #     pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
            #     gt_colors = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(gt_points), 1))
            #     pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)
            #     gt_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_gt, voxel_size=0.04)
            #     # o3d.io.write_point_cloud(f"points_gt_snn_{mode}.ply", pcd_gt)
            #
            #     o3d.visualization.draw_geometries([gt_vox])
            #     o3d.visualization.draw_geometries([pred_vox])

            # compute IoU at this timestep
            iou = iou_score(y_mean, yb)
            total_iou_t[t] += iou

        n_batches += 1

    total_iou_t /= n_batches
    total_spike_rate_t /= n_batches

    return total_iou_t, total_spike_rate_t


if __name__ == "__main__":
    from model import AE
    import torch
    from config import Config
    from dataset import PcdDataset
    from torch.utils.data import DataLoader
    from train import evaluate
    import matplotlib.pyplot as plt
    import numpy as np



    # ===== Font Configuration: Times New Roman + Larger Size =====
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 16          # Base font size (larger)
    plt.rcParams['axes.titlesize'] = 20     # Title font size
    plt.rcParams['axes.labelsize'] = 18     # Axis label font size
    plt.rcParams['xtick.labelsize'] = 14    # X-tick label size
    plt.rcParams['ytick.labelsize'] = 14    # Y-tick label size
    plt.rcParams['legend.fontsize'] = 14    # Legend font size
    plt.rcParams['figure.titlesize'] = 22   # Figure suptitle size
    # For PDF text rendering (editable text in PDF):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # =============================================================

    config = Config()
    device = config.device
    
    model = AE().to(device)
    model.load_state_dict(torch.load("best_model_new.pth", map_location=device))

    tr_dataset = PcdDataset(config=config, split="train")
    tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size_ann2snn, shuffle=True, collate_fn=collate_points)
    test_dataset = PcdDataset(config=config, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size_ann2snn_test, shuffle=False)

    test_loss, test_iou = evaluate(model, test_dataloader, config)
    print(f"Test Loss {test_loss:.4f} IoU {test_iou:.3f}")
    
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size_ann2snn_test, shuffle=False, collate_fn=collate_points)

    iou_curves = {}
    sr_curves = {}
    modes = ['max', '99.9%', 1/2, 1/3, 1/4, 1/5]
    modes1 = [1/5, 1/4, 1/3, 1/2, '99.9%','max']

    # modes = ['99.9%']
    color_cycle = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:cyan']
    color_cycle1 = ['tab:cyan', 'tab:purple', 'tab:orange', 'tab:green', 'tab:blue','tab:red']

    for mode, color in zip(modes, color_cycle):
        # snn_model = convert_ann_to_snn(model, tr_dataloader, device, mode)
        snn_model = load_fine_tuned_snn("MFT_best_snn_model_"+ str(mode)+".pth",device,mode)
        # print(snn_model)
        Smonitor = monitor.OutputMonitor(snn_model, neuron.IFNode)
        iou_curve, spike_rate_curve = evaluate_snn_over_time(Smonitor,snn_model, test_dataloader, device, mode, config.snn_steps)
        print(f"Final IoU {iou_curve[-1]:.3f}, Spike Rate {spike_rate_curve[-1]:.3f}")
        iou_curves[str(mode)] = iou_curve
        sr_curves[str(mode)] = spike_rate_curve

    # Combined Plot
    plt.figure(figsize=(7,5))
    # 添加 ANN 基准线
    plt.axhline(y=test_iou, color='black', linestyle='--', linewidth=2, label=f'ANN baseline ({test_iou:.4f})')
    for mode, color in zip(modes1, color_cycle1):
        plt.plot(np.arange(1, config.snn_steps + 1), iou_curves[str(mode)], marker='o', color=color, label=f"mode: {mode}")
    plt.xlabel('Time step')
    plt.ylabel('IoU')
    # plt.title('SNN IoU over time — multiple conversion modes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("snn_iou_comparison_all_modes.png", dpi=150)

    plt.figure(figsize=(7,5))
    # 添加 ANN 基准线
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2, label=f'ANN neural activity baseline ({1.0:.4f})')
    for mode, color in zip(modes1, color_cycle1):
        plt.plot(np.arange(1, config.snn_steps + 1), sr_curves[str(mode)], marker='o', color=color, label=f"mode: {mode}")
    plt.xlabel('Time step')
    plt.ylabel('Neural Activity')
    # plt.title('SNN spike rate over time — multiple conversion modes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("snn_sr_comparison_all_modes.png", dpi=150)

    plt.figure(figsize=(7,5))
    # 添加 ANN 基准线
    plt.axhline(y=test_iou, color='black', linestyle='--', linewidth=2, label=f'ANN IoU baseline ({test_iou:.4f})')
    plt.axvline(x=1, color='black', linestyle='--', linewidth=2, label=f'ANN neural activity baseline ({1.0:.4f})')
    for mode, color in zip(modes, color_cycle):
        plt.plot(sr_curves[str(mode)],iou_curves[str(mode)], marker='o', color=color, label=f"mode: {mode}")
    plt.ylabel('IoU')
    plt.xlabel('Neural Activity')
    # plt.title('SNN IoU vs spike rate over time — multiple conversion modes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("snn_sriou_comparison_all_modes.png", dpi=150)
