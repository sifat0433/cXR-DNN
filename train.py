import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from loss import dice_loss, iou_score
from utils import visualize_voxels_to_file, recover_depth_points_from_voxels


def train_epoch(model, dataloader, optimizer, config):
    device = config.device
    bce_pos_weight = config.bce_pos_weight
    dice_weight = config.dice_weight
    bce_weight = config.bce_weight
    model.train()
    total_loss, total_iou = 0.0, 0.0
    bce = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        pts = batch["points"].to(device, non_blocking=True)  # [B, N, 3]
        vox_gt = batch["vox"].to(device, non_blocking=True)  # [B, G, G, G]

        optimizer.zero_grad()
        vox_pred = model(pts)
        loss = bce_weight * bce(vox_pred, vox_gt) + (dice_weight * dice_loss(vox_pred, vox_gt))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Calculate IoU only once and reuse
        batch_iou = iou_score(vox_pred.detach(), vox_gt.detach()).item()
        total_iou += batch_iou
        
    n = len(dataloader)
    return total_loss / n, total_iou / n


def evaluate(model, dataloader, config):
    device = config.device
    bce_pos_weight = config.bce_pos_weight
    dice_weight = config.dice_weight
    bce_weight = config.bce_weight
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    bce = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        # cnt = 0
        for batch in dataloader:
            pts = batch["points"].to(device, non_blocking=True)  # [B, N, 3]
            vox_gt = batch["vox"].to(device, non_blocking=True)  # [B, G, G, G]

            vox_pred = model(pts)
            loss = bce_weight * bce(vox_pred, vox_gt) + (dice_weight * dice_loss(vox_pred, vox_gt))

            total_loss += loss.item()
            total_iou += iou_score(vox_pred, vox_gt).item()

            # for visualization
            # if cnt < 10:
            #     vox_pred_np = vox_pred.detach().cpu().numpy()[0]
            #     vox_gt_np = vox_gt.detach().cpu().numpy()[0]
            #     grid_size = vox_pred_np.shape[0]
            #     pred_points = recover_depth_points_from_voxels(vox_pred_np, grid_size)
            #     gt_points = recover_depth_points_from_voxels(vox_gt_np, grid_size)

            #     pcd_pred = o3d.geometry.PointCloud()
            #     pcd_pred.points = o3d.utility.Vector3dVector(pred_points)
            #     pred_colors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(pred_points), 1))
            #     pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
            #     o3d.io.write_point_cloud(f"points_pred_{cnt}.ply", pcd_pred)

            #     pcd_gt = o3d.geometry.PointCloud()
            #     pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
            #     gt_colors = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(gt_points), 1))
            #     pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)
            #     o3d.io.write_point_cloud(f"points_gt_{cnt}.ply", pcd_gt)
            #     cnt += 1

    n = len(dataloader)
    return total_loss / n, total_iou / n


if __name__ == "__main__":
    from config import Config
    from model import AE
    from dataset import PcdDataset
    from torch.utils.data import DataLoader
    from utils import estimate_pos_weight

    config = Config()
    model = AE().to(config.device)

    tr_dataset = PcdDataset(config=config, split="train")
    # Optimize DataLoader: use multiple workers and pin_memory for faster data loading
    tr_dataloader = DataLoader(
        tr_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,  # Parallel data loading
        pin_memory=True if config.device == "cuda" else False,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )

    if config.bce_pos_weight is None:
        config.bce_pos_weight = estimate_pos_weight(tr_dataloader)
        print(f"Estimated pos_weight ≈ {config.bce_pos_weight:.2f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    val_dataset = PcdDataset(config=config, split="val")
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size_val_test, 
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
        persistent_workers=True
    )

    best_iou = 0.0
    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_iou = train_epoch(model, tr_dataloader, optimizer, config)
        print(f"Epoch {epoch:03d}/{config.epochs} | Train Loss {tr_loss:.4f} IoU {tr_iou:.3f}")
        val_loss, val_iou = evaluate(model, val_dataloader, config)
        print(f"Epoch {epoch:03d}/{config.epochs} | Val Loss {val_loss:.4f} IoU {val_iou:.3f}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"best_model.pth")
            torch.save(model.encoder.state_dict(), f"best_encoder_model.pth")
            torch.save(model.decoder.state_dict(), f"best_decoder_model.pth")
            print(f"Saved best model to best_model.pth")