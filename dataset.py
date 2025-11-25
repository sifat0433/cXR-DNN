import os
from torch.utils.data import Dataset
from config import Config
import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PcdDataset(Dataset):
    def __init__(self, config: Config, split:str):
        assert split in ["train", "val", "test"], "Invalid split, must be 'train', 'val' or 'test'"
        self.file_list = open(os.path.join(config.data_split_path, f"{split}.txt")).readlines()
        self.file_list = [os.path.join(config.data_path, f.strip()) for f in self.file_list]
        print(f"Found {len(self.file_list)} {split} files")

        self.n_points = config.n_points
        self.grid_size = config.grid_size
        self.latent_dim = config.latent_dim

    def normalize_to_unit_cube(self, points, eps=1e-8):
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        scale = np.maximum(pmax - pmin, eps)
        return (points - pmin) / scale

    def sample_points(self, points, colors):
        N = len(points)
        if N == 0:
            raise ValueError("Empty point cloud.")
        replace = N < self.n_points
        sel = np.random.choice(N, self.n_points, replace=replace)
        return points[sel], colors[sel]

    def voxelize_unit_cube(self, points):
        """
        Convert normalized depth points into an occlusion-aware voxel grid.
        For every (x, y) column along the depth axis (z), once a surface point is
        encountered, the entire occluded region behind it is marked as 1.
        """
        idx = np.clip((points * (self.grid_size - 1)).astype(np.int32), 0, self.grid_size - 1)
        vox = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        vox[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

        # Fill occluded regions: cumulative sum along depth axis turns every column
        # behind the first hit to 1 while keeping empty columns at 0.
        occluded = np.cumsum(vox, axis=2)
        occluded[occluded > 0] = 1.0
        return occluded

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.file_list[idx])
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)

        assert points.shape[0] == colors.shape[0], "Points and colors must have the same number of points"

        # normalize to unit cube
        points = self.normalize_to_unit_cube(points)
        
        # voxelize
        vox = self.voxelize_unit_cube(points)

        # sample points
        points, colors = self.sample_points(points, colors)
        
        return {"points": points, "colors": colors, "vox": vox}


if __name__ == "__main__":    
    config = Config()
    dataset = PcdDataset(config=config, split="train")
    print(len(dataset))

    from torch.utils.data import DataLoader
    from utils import visualize_voxels_to_file
    from utils import recover_depth_points_from_voxels
    grid_size = config.grid_size

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for data in dataloader:
        print(data["points"].shape) # [N, n_points, 3]
        print(data["colors"].shape) # [N, n_points, 3]
        print(data["vox"].shape) # [N, G, G, G]
        
        # Example: visualize voxels
        vox_np = data["vox"][0].numpy()  # Take first item from batch
        visualize_voxels_to_file(vox_np, "voxel_vis.ply", format='ply')
        # visualize_voxels_to_file(vox_np, "voxel_vis.obj", format='obj')
        # visualize_voxels_to_file(vox_np, "voxel_vis.png", format='png')
        points = recover_depth_points_from_voxels(vox_np, grid_size)
        
        # save points to ply
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud("points.ply", pcd)

        # visualize gt points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data["points"][0])
        o3d.io.write_point_cloud("gt_points.ply", pcd)

        break