from dataclasses import dataclass
import torch


@dataclass
class Config:
    data_path: str = "/mnt/cluster/workspaces/lichenyan/cXR-DNN/data/ply"
    data_split_path: str = "/mnt/cluster/workspaces/lichenyan/cXR-DNN/data/split"
    grid_size: int = 64
    n_points: int = 4096

    num_workers: int = 4
    batch_size: int = 128
    betch_size_val_test: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim: int = 128
    
    epochs: int = 100
    lr: float = 1e-2

    dice_weight: float = 0.7
    bce_weight: float = 0.3
    bce_pos_weight: float = 50.0