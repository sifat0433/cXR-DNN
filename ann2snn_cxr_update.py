# ------------------------------------------------------------
# convert_and_test_snn_streaming.py


import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open3d as o3d

import matplotlib.pyplot as plt

# Optional: reduce CUDA fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from spikingjelly.activation_based import ann2snn, functional

# ----------------------------
# Config
# ----------------------------
DATA_GLOB   = "all/frame*.ply"   # e.g., frame000.ply ... frame114.ply
GRID_SIZE   = 64
N_POINTS    = 4096               # sampled points per cloud
LATENT_DIM  = 128
EPOCHS      = 1500               # (unused here; kept for parity)
LR          = 1e-3               # (unused here; kept for parity)
DICE_W, BCE_W = 0.7, 0.3         # (unused here; kept for parity)
CHECKPOINT  = "voxel_autoencoder_pointnet3dconv.pth"
DEVICE      = "cuda"  # if torch.cuda.is_available() else "cpu"
SEED        = 42
voxel_size  = 0.04   # 1.0 / GRID_SIZE

# Conversion/Test controls
BATCH_CONVERTER = 16      # streaming batch size for converter dataloader
BATCH_TEST      = 8      # test batch size (SNN can be memory hungry)
T_STEPS         = 10      # SNN time steps for evaluation


mode_max_accs = []
mode_robust_accs = []
mode_two_accs = []
mode_three_accs = []
mode_four_accs = []
mode_five_accs = []


torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# Utils
# ----------------------------
def normalize_to_unit_cube(points, eps=1e-8):
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    scale = np.maximum(pmax - pmin, eps)
    return (points - pmin) / scale

def sample_points(points01, n_points=N_POINTS):
    N = len(points01)
    if N == 0:
        raise ValueError("Empty point cloud.")
    replace = N < n_points
    sel = np.random.choice(N, n_points, replace=replace)
    return points01[sel]

def voxelize_unit_cube(points01, grid_size=GRID_SIZE):
    idx = np.clip((points01 * (grid_size - 1)).astype(np.int32), 0, grid_size - 1)
    vox = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    vox[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return vox

@torch.no_grad()
def iou_score(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    inter = (preds * targets).sum(dim=[1,2,3,4])
    union = (preds + targets - preds*targets).sum(dim=[1,2,3,4])
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

# ----------------------------
# Streaming Datasets
# ----------------------------
class PointsOnlyDataset(Dataset):
    """
    only inputs [N,3] for the ann2snn Converter.
    """
    def __init__(self, files, n_points=N_POINTS):
        self.o3d = o3d
        self.files = files
        self.n_points = n_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        pcd = self.o3d.io.read_point_cloud(f)
        pts = np.asarray(pcd.points, dtype=np.float32)
        pts01 = normalize_to_unit_cube(pts)
        sampled = sample_points(pts01, N_POINTS)      # (N,3) numpy
        x = torch.from_numpy(sampled).float()    # [N,3]  <-- not [3,N] or [N]
        return x, 0


class PointsAndVoxelDataset(Dataset):
    """
input points [N,3], target voxels [1,G,G,G] for evaluation.
    """
    def __init__(self, files, n_points=N_POINTS, grid_size=GRID_SIZE):
        self.o3d = o3d
        self.files = files
        self.n_points = n_points
        self.grid_size = grid_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        pcd = self.o3d.io.read_point_cloud(f)
        pts = np.asarray(pcd.points, dtype=np.float32)
        pts01 = normalize_to_unit_cube(pts)

        # targets from full cloud
        vox = voxelize_unit_cube(pts01, grid_size=self.grid_size)     # [G,G,G]
        # inputs as sampled points
        sampled = sample_points(pts01, self.n_points)                 # [N,3]

        x = torch.from_numpy(sampled).float()                         # [N,3]
        y = torch.from_numpy(vox).float().unsqueeze(0)                # [1,G,G,G]
        return x, y

# ----------------------------
# Model
# ----------------------------
class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):  # [B, N, 3]
        x_flat = x.reshape(-1, 3)                     # [B*N, 3]
        feat_flat = self.mlp(x_flat)            # [B*N, latent]
        feat = feat_flat.view(x.shape[0], -1, feat_flat.shape[-1])
        global_feat = torch.mean(feat, dim=1)         # [B, latent_dim]
        return global_feat




class Conv3DDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, grid_size=GRID_SIZE):
        super().__init__()
        assert grid_size % 8 == 0, "GRID_SIZE should be multiple of 8."
        self.seed_size = 8
        seed_channels = 64
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, self.seed_size**3 * seed_channels), nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(seed_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1)
        )
    def forward(self, z):  # [B,latent]
        seed = self.fc(z)
        seed = seed.view(-1, 64, self.seed_size, self.seed_size, self.seed_size)
        logits = self.deconv(seed)
        return logits

class AE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, grid_size=GRID_SIZE):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = Conv3DDecoder(latent_dim, grid_size)
    def forward(self, x_points):  # [B,N,3]
        z = self.encoder(x_points)
        logits = self.decoder(z)
        return logits



# ----------------------------
# SNN utilities
# ----------------------------

def collate_points(batch):
    """
    batch: list of (x, y) pairs, where x: [N, 3], y: dummy
    Returns: (X [B, N, 3], Y [B]) dummy tensor
    """
    xs = []
    ys = []
    for item in batch:
        x, y = item
        xs.append(x.contiguous())
        ys.append(torch.tensor(y, dtype=torch.float32))
    X = torch.stack(xs, dim=0)  # [B, N, 3]
    Y = torch.stack(ys, dim=0)  # [B]
    return X, Y


def collate_points_vox(batch):
    """
    batch: list of (x, y)
      x: [N,3], y: [1,G,G,G]
    Returns: (X [B,N,3], Y [B,1,G,G,G])
    """
    xs, ys = [], []
    for x, y in batch:
        xs.append(torch.as_tensor(x, dtype=torch.float32))
        ys.append(torch.as_tensor(y, dtype=torch.float32))
    X = torch.stack(xs, dim=0)  # [B,N,3]
    Y = torch.stack(ys, dim=0)  # [B,1,G,G,G]

    return X, Y

@torch.no_grad()
def evaluate_snn_over_time_and_save(snn_model, test_loader, device, mode, T=T_STEPS, amp_dtype=None):
    """
    Evaluates a converted SNN over multiple timesteps and saves IoU results and model weights per mode.
    """
    snn_model.eval()
    print(f"[INFO] Evaluating SNN for {T} timesteps (mode={mode})...")

    total_iou_t = np.zeros(T, dtype=np.float32)
    n_batches = 0
    for xb_cpu, yb_cpu in test_loader:
        xb = xb_cpu.to(device, non_blocking=True)  # [B, N, 3]
        yb = yb_cpu.to(device, non_blocking=True)  # [B, 1, G, G, G]

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

    # ---------- Save and Plot ----------
    safe_mode = str(mode).replace('%', 'pct').replace('/', '_')


    # Save results
    model_save_path = f"snn_model_mode_{safe_mode}.pth"

    torch.save({
        "snn_state_dict": snn_model.state_dict(),
        "arch": "AE_SNN",
        "mode": mode,
        "timesteps": T,
    }, model_save_path)

    print(f"[OK] Saved SNN model weights to: {model_save_path}")

    return total_iou_t

def run_multi_mode_conversion_and_eval(model, train_files, test_files, device, T=T_STEPS):
    """
    Runs ANN2SNN conversion and evaluation across multiple conversion modes.
    Saves per-mode IoU curves and a combined plot.
    """
    from torch.utils.data import DataLoader

    modes = ['max', '99.9%', 1/2, 1/3, 1/4, 1/5]
    color_cycle = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:cyan']
    all_curves = {}

    converter_ds = PointsOnlyDataset(train_files, n_points=N_POINTS)
    test_ds = PointsAndVoxelDataset(test_files, n_points=N_POINTS, grid_size=GRID_SIZE)

    converter_loader = DataLoader(
        converter_ds, batch_size=BATCH_CONVERTER, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_points
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_TEST, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_points_vox
    )

    for mode, color in zip(modes, color_cycle):
        print(f"\n[INFO] === Converting & Evaluating mode={mode} ===")

        converter = ann2snn.Converter(dataloader=converter_loader, device=str(device), mode=mode)
        snn_model = converter(model).eval()

        iou_curve = evaluate_snn_over_time_and_save(snn_model, test_loader, device, mode=mode, T=T)
        all_curves[str(mode)] = iou_curve

    # Combined Plot
    plt.figure(figsize=(7,5))
    for mode, color in zip(modes, color_cycle):
        plt.plot(np.arange(1, T + 1), all_curves[str(mode)], marker='o', color=color, label=f"mode: {mode}")
    plt.xlabel('Time step')
    plt.ylabel('IoU')
    plt.title('SNN IoU over time — multiple conversion modes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("snn_iou_comparison_all_modes.png", dpi=150)
    plt.show()
    print("[OK] Saved combined plot: snn_iou_comparison_all_modes.png")

    torch.save(all_curves, "snn_all_modes_iou_curves.pth")
    print("[OK] Saved all IoU curves to: snn_all_modes_iou_curves.pth")

    return all_curves



# ----------------------------
# Main

def main():
    # Split files
    files = sorted(glob.glob(DATA_GLOB))
    assert len(files) > 1, f"No files found with pattern {DATA_GLOB}"
    random.shuffle(files)
    n_train = int(0.8 * len(files))
    train_files = files[:n_train]
    test_files  = files[n_train:]
    print(f"[INFO] Found {len(files)} files → train {len(train_files)} / test {len(test_files)}")

    # Device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Rebuild ANN and load checkpoint
    model = AE(latent_dim=LATENT_DIM, grid_size=GRID_SIZE).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[INFO] Loaded model from checkpoint: {CHECKPOINT}")
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] Loaded raw state_dict from checkpoint: {CHECKPOINT}")
    model.eval()

    print("[INFO] Starting multi-mode conversion and evaluation...")
    all_curves = run_multi_mode_conversion_and_eval(model, train_files, test_files, device, T=T_STEPS)


if __name__ == "__main__":
    main()

