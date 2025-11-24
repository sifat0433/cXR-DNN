from spikingjelly.activation_based import ann2snn, functional
import torch
from loss import iou_score
import numpy as np


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

@torch.no_grad()
def evaluate_snn_over_time(snn_model, test_loader, device, mode, T):
    snn_model.eval()
    print(f"Evaluating SNN for {T} timesteps (mode={mode})...")

    total_iou_t = np.zeros(T, dtype=np.float32)
    n_batches = 0
    for xb_cpu, yb_cpu in test_loader:
        xb = xb_cpu.to(device, non_blocking=True)  # [B, N, 3]
        yb = yb_cpu.to(device, non_blocking=True)  # [B, G, G, G]

        functional.reset_net(snn_model)
        y_sum = 0
        for t in range(T):
            y_t = snn_model(xb)
            y_sum += y_t
            y_mean = y_sum / (t + 1)  # running mean over timesteps

            # compute IoU at this timestep
            iou = iou_score(y_mean, yb)
            total_iou_t[t] += iou

        n_batches += 1

    total_iou_t /= n_batches

    return total_iou_t


if __name__ == "__main__":
    from model import AE
    import torch
    from config import Config
    from dataset import PcdDataset
    from torch.utils.data import DataLoader
    from train import evaluate
    import matplotlib.pyplot as plt
    import numpy as np
    config = Config()
    device = config.device
    
    model = AE().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    tr_dataset = PcdDataset(config=config, split="train")
    tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size_ann2snn, shuffle=True, collate_fn=collate_points)
    test_dataset = PcdDataset(config=config, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size_ann2snn_test, shuffle=False)

    test_loss, test_iou = evaluate(model, test_dataloader, config)
    print(f"Test Loss {test_loss:.4f} IoU {test_iou:.3f}")
    
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size_ann2snn_test, shuffle=False, collate_fn=collate_points)

    all_curves = {}
    modes = ['max', '99.9%', 1/2, 1/3, 1/4, 1/5]
    color_cycle = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:cyan']
    for mode, color in zip(modes, color_cycle):
        snn_model = convert_ann_to_snn(model, tr_dataloader, device, mode)
        # print(snn_model)
        iou_curve = evaluate_snn_over_time(snn_model, test_dataloader, device, mode, config.snn_steps)
        print(f"Final IoU {iou_curve[-1]:.3f}")
        all_curves[str(mode)] = iou_curve

    # Combined Plot
    plt.figure(figsize=(7,5))
    # 添加 ANN 基准线
    plt.axhline(y=test_iou, color='black', linestyle='--', linewidth=2, label=f'ANN baseline ({test_iou:.4f})')
    for mode, color in zip(modes, color_cycle):
        plt.plot(np.arange(1, config.snn_steps + 1), all_curves[str(mode)], marker='o', color=color, label=f"mode: {mode}")
    plt.xlabel('Time step')
    plt.ylabel('IoU')
    plt.title('SNN IoU over time — multiple conversion modes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("snn_iou_comparison_all_modes.png", dpi=150)