import os
import open3d as o3d
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Constants
DATA_URL = ""
DATA_DIR = "./redwood_pc"
DATA_PATH = os.path.join(DATA_DIR, "00033.ply")
VOXEL_SIZE = 0.05
GRID_SIZE = 64
LATENT_DIM = 64
EPOCHS = 50

# # Download Redwood point cloud
# os.makedirs(DATA_DIR, exist_ok=True)
# if not os.path.isfile(DATA_PATH):
#     print("Downloading Redwood point cloud...")
#     urllib.request.urlretrieve(DATA_URL, DATA_PATH)

# Load point cloud
pcd = o3d.io.read_point_cloud(DATA_PATH)
points = np.asarray(pcd.points)



# Normalize points
points -= points.mean(axis=0)
points /= np.std(points, axis=0)

# Voxelization
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
voxels = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
for voxel in voxel_grid.get_voxels():
    idx = voxel.grid_index
    if all(0 <= i < GRID_SIZE for i in idx):
        voxels[idx[0], idx[1], idx[2]] = 1.0

        # voxels = np.unique(np.asarray([voxel_grid.get_voxel(pt) for pt in points]), axis=0)

# Sample points
N = 1024
if len(points) >= N:
    idx = np.random.choice(len(points), N, replace=False)
else:
    idx = np.random.choice(len(points), N, replace=True)
sampled_points = points[idx]

# Convert to tensors
x_input = torch.tensor(sampled_points, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
y_voxel = torch.tensor(voxels, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, L, H, W]

# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        x = self.mlp(x)  # [B, N, latent_dim]
        return torch.max(x, dim=1)[0]  # Global feature [B, latent_dim]

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE)):
        super().__init__()
        self.out_shape = out_shape
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(out_shape)),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.net(x)
        return x.view(-1, 1, *self.out_shape)

# Model
encoder = Encoder(LATENT_DIM)
decoder = Decoder(LATENT_DIM)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
criterion = nn.BCELoss()

# Training
print("Training model...")
for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()
    code = encoder(x_input)
    pred = decoder(code)
    loss = criterion(pred, y_voxel)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")


def numpy_to_voxelgrid(voxel_data, voxel_size=0.05, offset=np.array([0, 0, 0])):
    """
    Convert a binary 3D NumPy voxel grid into an Open3D VoxelGrid.
    The offset shifts the entire grid in space (used instead of .translate()).
    """
    occupied = np.argwhere(voxel_data > 0.5)
    voxel_centers = offset + occupied * voxel_size + voxel_size / 2.0

    # Create point cloud from voxel centers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)

    # Convert to voxel grid
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)


# 6. Report Compression Ratio
original_size = np.prod(points.shape)
compressed_size = LATENT_DIM
print(f"\nCompression Ratio (Original:Compressed) = {original_size}:{compressed_size} ≈ {original_size/compressed_size:.2f}x")

# Visualization
pred_vox = pred.detach().squeeze().numpy()
gt_vox = y_voxel.squeeze().numpy()
#
# print(gt_vox)

# Create voxel grid objects from binary data
gt_voxel_o3d = numpy_to_voxelgrid(gt_vox)
recon_voxel_o3d = numpy_to_voxelgrid(pred_vox)


# Display point cloud + voxel grids (side-by-side)
print("Displaying PCD + Ground Truth VoxelGrid + Reconstructed VoxelGrid...")
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([gt_voxel_o3d])
o3d.visualization.draw_geometries([recon_voxel_o3d])
