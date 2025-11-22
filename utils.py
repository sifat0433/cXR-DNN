import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def visualize_voxels_to_file(vox, output_path, format='ply', normalize=True):
    """
    Visualize voxels by saving to a file that can be opened with external apps.
    
    Args:
        vox: numpy array of shape (G, G, G) or (1, G, G, G) - voxel grid
        output_path: str or Path - output file path
        format: str - output format: 'ply', 'obj', or 'png'
        normalize: bool - if True, normalize voxel coordinates to [0, 1]
    
    Supported formats:
        - 'ply': Point cloud format (can be opened with CloudCompare, MeshLab, Blender, etc.)
        - 'obj': Mesh format (can be opened with MeshLab, Blender, ParaView, etc.)
        - 'png': Image slices showing XY, XZ, YZ projections (can be viewed with any image viewer)
    """
    # Handle batch dimension
    if vox.ndim == 4:
        vox = vox[0]  # Take first item if batch
    
    assert vox.ndim == 3, f"Expected 3D voxel array, got shape {vox.shape}"
    
    # Normalize voxel coordinates to [0, 1] if requested
    if normalize:
        # Convert voxel indices to normalized coordinates
        grid_size = vox.shape[0]
        x, y, z = np.where(vox > 0.5)  # Get occupied voxel indices
        if len(x) == 0:
            print(f"[WARNING] No occupied voxels found in the grid")
            return
        
        # Normalize to [0, 1]
        points = np.stack([
            x.astype(np.float32) / (grid_size - 1),
            y.astype(np.float32) / (grid_size - 1),
            z.astype(np.float32) / (grid_size - 1)
        ], axis=1)
    else:
        # Use raw voxel indices
        x, y, z = np.where(vox > 0.5)
        if len(x) == 0:
            print(f"[WARNING] No occupied voxels found in the grid")
            return
        points = np.stack([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)], axis=1)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'ply':
        # Save as PLY point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Add white color for all points
        colors = np.ones((len(points), 3), dtype=np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"[INFO] Saved voxel visualization as PLY: {output_path}")
        print(f"       Can be opened with: CloudCompare, MeshLab, Blender, ParaView")
        
    elif format.lower() == 'obj':
        # Save as OBJ mesh (simple cube representation)
        with open(output_path, 'w') as f:
            f.write("# OBJ file generated from voxel grid\n")
            # Write vertices (voxel centers)
            for p in points:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            # Write simple point cloud (no faces)
            # For mesh representation, you could add cube faces here
            print(f"[INFO] Saved voxel visualization as OBJ: {output_path}")
            print(f"       Can be opened with: MeshLab, Blender, ParaView, MeshLab")
            
    elif format.lower() == 'png':
        # Save as PNG image slices (projections)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY projection (sum along Z axis)
        proj_xy = np.sum(vox, axis=2)
        axes[0].imshow(proj_xy, cmap='hot', origin='lower')
        axes[0].set_title('XY Projection (Z-axis sum)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # XZ projection (sum along Y axis)
        proj_xz = np.sum(vox, axis=1)
        axes[1].imshow(proj_xz, cmap='hot', origin='lower')
        axes[1].set_title('XZ Projection (Y-axis sum)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        
        # YZ projection (sum along X axis)
        proj_yz = np.sum(vox, axis=0)
        axes[2].imshow(proj_yz, cmap='hot', origin='lower')
        axes[2].set_title('YZ Projection (X-axis sum)')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved voxel visualization as PNG: {output_path}")
        print(f"       Can be viewed with: any image viewer")
        
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats: 'ply', 'obj', 'png'")

def estimate_pos_weight(dataloader):
    """
    Estimate positive weight for binary classification loss.
    
    Args:
        dataloader: DataLoader instance
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for DataLoader
    
    Returns:
        float: Positive weight (clamped between 1.0 and 50.0)
    """
    
    # Store results in lists
    pos_list = []
    neg_list = []
    
    # Iterate through DataLoader and collect results
    for batch in dataloader:
        vox = batch["vox"]
        # Handle both numpy and torch tensors
        if isinstance(vox, torch.Tensor):
            p = vox.sum().item()
            n = vox.numel() - p
        else:
            p = float(vox.sum())
            n = float(vox.size - p)
        
        pos_list.append(p)
        neg_list.append(n)
    
    # Sum all values from lists
    pos = sum(pos_list)
    neg = sum(neg_list)
    
    # Clamp to avoid extremes
    w = max(1.0, min(neg / max(pos, 1.0), 50.0))
    return float(w)