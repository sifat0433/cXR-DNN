import os
import glob
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config
# ----------------------------
DATA_GLOB   = "frame*.ply"  # e.g., frame000.ply ... frame114.ply
GRID_SIZE   = 64
N_POINTS    = 2048          # sampled points per cloud (increase if you have GPU RAM)
LATENT_DIM  = 128
EPOCHS      = 200
LR          = 1e-4
POS_WEIGHT  = 200.0          # class-imbalance weight for occupied voxels
CHECKPOINT  = "voxel_autoencoder_pointnet3dconv.pth"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
voxel_size = 0.04#1.0 / GRID_SIZE

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Utils: normalization, voxelization, visualization
# ----------------------------
def normalize_to_unit_cube(points, eps=1e-8):
    """
    Normalize Nx3 points to [0,1]^3 by min-max per axis.
    More stable and consistent than relying on Open3D VoxelGrid indices.
    """
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    scale = np.maximum(pmax - pmin, eps)
    normed = (points - pmin) / scale
    return normed

def voxelize_unit_cube(points01, grid_size=GRID_SIZE):
    """
    points01 are Nx3 in [0,1]^3. Bin to grid_size^3 occupancy grid.
    """
    idx = np.clip((points01 * (grid_size - 1)).astype(np.int32), 0, grid_size - 1)
    vox = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    vox[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return vox

def sample_points(points, n_points=N_POINTS):
    """
    Uniformly sample n_points with replacement if needed.
    """
    N = len(points)
    if N == 0:
        raise ValueError("Empty point cloud.")
    if N >= n_points:
        sel = np.random.choice(N, n_points, replace=False)
    else:
        sel = np.random.choice(N, n_points, replace=True)
    return points[sel]

def recons_voxel(voxel_grid_np, color=[1.0, 0.5, 0.0], voxel_size=voxel_size, offset=np.array([0.0, 0.0, 0.0])):
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

# ----------------------------
# Data loading
# ----------------------------

def load_dataset(file_list, grid_size=GRID_SIZE, n_points=N_POINTS):
    data = []
    for f in file_list:
        pcd = o3d.io.read_point_cloud(f)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] == 0:
            continue
        # Normalize to unit cube
        pts01 = normalize_to_unit_cube(pts)

        # Build voxel GT from full (normalized) cloud
        vox = voxelize_unit_cube(pts01, grid_size=grid_size)

        # Sample per-point input (still in [0,1]^3)
        sampled = sample_points(pts01, n_points=n_points)

        data.append((sampled.astype(np.float32), vox.astype(np.float32), f,pcd))
    return data

# ----------------------------
# Model: PointNet-style Encoder + 3D Conv Decoder
# ----------------------------
class PointNetEncoder(nn.Module):
    """
    Per-point MLP -> global max-pool (PointNet-style).
    Input:  B x N x 3
    Output: B x LATENT_DIM
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):  # x: [B, N, 3]
        feat = self.mlp(x)             # [B, N, latent_dim]
        global_feat, _ = torch.max(feat, dim=1)  # [B, latent_dim]
        return global_feat

class Conv3DDecoder(nn.Module):
    """
    Latent -> 8x8x8 seed volume -> upsample via ConvTranspose3d to 64^3.
    Output are logits (no sigmoid inside). Use BCEWithLogitsLoss.
    """
    def __init__(self, latent_dim=LATENT_DIM, grid_size=GRID_SIZE):
        super().__init__()
        assert grid_size % 8 == 0, "GRID_SIZE should be multiple of 8 (e.g., 32, 64)."
        self.seed_size = 8
        self.gs = grid_size

        seed_channels = 64
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, self.seed_size * self.seed_size * self.seed_size * seed_channels), nn.ReLU(inplace=True)
        )

        # Upsample 8 -> 16 -> 32 -> 64
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(seed_channels, 64, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),            # 16->32
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),            # 32->64
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),

            nn.Conv3d(16, 1, kernel_size=1)  # logits output: [B,1,64,64,64]
        )

    def forward(self, z):  # z: [B, latent_dim]
        B = z.shape[0]
        seed = self.fc(z)
        seed = seed.view(B, 64, self.seed_size, self.seed_size, self.seed_size)
        logits = self.deconv(seed)  # [B,1,gs,gs,gs]
        return logits

# Combined model wrapper
class AE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, grid_size=GRID_SIZE):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = Conv3DDecoder(latent_dim, grid_size)

    def forward(self, x_points):  # x_points: [B,N,3]
        z = self.encoder(x_points)
        logits = self.decoder(z)
        return logits

# ----------------------------
# Losses / Metrics
# ----------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()

def iou_score(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    inter = (preds * targets).sum(dim=[1,2,3,4])
    union = (preds + targets - preds*targets).sum(dim=[1,2,3,4])
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

# ----------------------------
# Train / Eval
# ----------------------------
def train_epoch(model, data, optimizer, bce_pos_weight, use_dice=True):
    model.train()
    total_loss, total_iou = 0.0, 0.0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight], device=DEVICE))
    dice = DiceLoss()

    for pts, vox, _, _ in data:
        x = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)           # [1,N,3]
        y = torch.tensor(vox, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)  # [1,1,G,G,G]

        optimizer.zero_grad()
        logits = model(x)
        loss_bce = bce(logits, y)
        loss = loss_bce + (dice(logits, y) if use_dice else 0.0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += iou_score(logits.detach(), y.detach())

    n = max(len(data), 1)
    return total_loss / n, total_iou / n

@torch.no_grad()
def eval_epoch(model, data):
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
    dice = DiceLoss()

    for pts, vox, _,_ in data:
        x = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        y = torch.tensor(vox, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits = model(x)
        loss = bce(logits, y) + dice(logits, y)
        total_loss += loss.item()
        total_iou  += iou_score(logits, y)

    n = max(len(data), 1)
    return total_loss / n, total_iou / n

# ----------------------------
# Main
# ----------------------------
def main():
    files = sorted(glob.glob(DATA_GLOB))
    assert len(files) > 1, f"No files found with pattern {DATA_GLOB}"
    n_train = int(0.8 * len(files))
    train_files = files[:n_train]
    test_files  = files[n_train:]

    print(f"Found {len(files)} files → train {len(train_files)} / test {len(test_files)}")

    train_data = load_dataset(train_files, grid_size=GRID_SIZE, n_points=N_POINTS)
    test_data  = load_dataset(test_files,  grid_size=GRID_SIZE, n_points=N_POINTS)
    print(f"Prepared {len(train_data)} train samples, {len(test_data)} test samples.")

    model = AE(latent_dim=LATENT_DIM, grid_size=GRID_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # If checkpoint exists, load it (skip training if you like)
    if os.path.isfile(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        print(f"Loaded checkpoint: {CHECKPOINT}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_iou = train_epoch(model, train_data, optimizer, POS_WEIGHT, use_dice=True)
        print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss {tr_loss:.4f} IoU {tr_iou:.3f}")

    # Save model
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "config": {
            "GRID_SIZE": GRID_SIZE,
            "N_POINTS": N_POINTS,
            "LATENT_DIM": LATENT_DIM
        }
    }, CHECKPOINT)
    print(f"Saved checkpoint to {CHECKPOINT}")

    # ------------------------
    # Visualize a few test samples in Open3D
    # ------------------------
    # We'll show: input PCD (green), GT voxels (orange), Pred voxels (cyan)

    for i, (pts, vox, fname,or_pts) in enumerate(test_data):
        print(f"Visualizing {fname} (test sample {i+1})")

        x = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0, 0]  # [G,G,G]
        pred_bin = (probs >= 0.5).astype(np.float32)

        # Open3D point cloud for input (already normalized to [0,1]^3)
        # pcd_in = o3d.geometry.PointCloud()
        # pcd_in.points = o3d.utility.Vector3dVector(or_pts)
        # pcd_in.paint_uniform_color([0.0, 0.8, 0.0])  # green

        # Convert voxels to colored point clouds (centers)
        gt_vox = recons_voxel(vox, voxel_size=voxel_size, offset=np.array([1.2, 0.0, 0.0]))
        pred_vox = recons_voxel(pred_bin, voxel_size=voxel_size, offset=np.array([-1.2, 0.0, 0.0]))

        o3d.visualization.draw_geometries([or_pts])
        o3d.visualization.draw_geometries([gt_vox])
        o3d.visualization.draw_geometries([pred_vox])

if __name__ == "__main__":
    main()
