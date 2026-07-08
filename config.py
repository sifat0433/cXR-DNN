from dataclasses import dataclass
import torch


@dataclass
class Config:
    data_path: str = "/home/gainpc2/Documents/pycharm5090/Gain/ply/cXR-DNN-main/all" 
    data_split_path: str = "/home/gainpc2/Documents/pycharm5090/Gain/ply/cXR-DNN-main/split"
    grid_size: int = 64
    n_points: int = 4096

    num_workers: int = 4
    batch_size: int = 16
    batch_size_val_test: int = 1
    batch_size_ann2snn: int = 32
    batch_size_ann2snn_test: int = 1
    batch_size_snn_train: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    snn_steps: int = 9

    latent_dim: int = 128
    
    epochs: int = 1
    lr: float = 1e-3

    dice_weight: float = 0.7
    bce_weight: float = 0.3
    bce_pos_weight: float = 33.0
