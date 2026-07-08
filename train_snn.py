"""
SNN Model Fine-tuning Training Script

Fine-tune the converted SNN model on real data to improve performance.
Optimize using time-accumulated outputs.
"""

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import functional , monitor, neuron
# from sympy.plotting.pygletplot.util import model_to_screen

from loss import dice_loss, iou_score
from config import Config
from dataset import PcdDataset
from torch.utils.data import DataLoader
from ann2snn import collate_points


def train_snn_epoch(Smonitor,snn_model, dataloader, optimizer, config, epoch=1, total_epochs=50, log_interval=10):
    """
    Train SNN model for one epoch
    
    Args:
        snn_model: Converted SNN model
        dataloader: Data loader
        optimizer: Optimizer
        config: Config object
        epoch: Current epoch number
        total_epochs: Total number of epochs
        log_interval: Number of batches between logging
        
    Returns:
        Average loss and average IoU
    """
    device = config.device
    dice_weight = config.dice_weight
    bce_weight = config.bce_weight
    snn_model.train()
    
    total_loss = 0.0
    total_iou = 0.0
    bce = nn.BCEWithLogitsLoss()
    T = config.snn_steps
    batch_count = 0
    
    for batch_idx, (pts, vox_gt) in enumerate(dataloader, 1):
        pts = pts.to(device, non_blocking=True)  # [B, N, 3]
        vox_gt = vox_gt.to(device, non_blocking=True)  # [B, G, G, G]
        
        optimizer.zero_grad()
        
        # Reset network state
        functional.reset_net(snn_model)
        
        # Run through multiple timesteps and accumulate outputs
        y_sum = 0
        total_spikes = 0
        total_neu = 0
        for t in range(T):
            y_t = snn_model(pts)  # [B, G, G, G]
            y_sum += y_t


        for rec in Smonitor.records:
            total_spikes += (rec > 0).sum().item()
            total_neu += rec.numel()

        Smonitor.clear_recorded_data()

        if epoch < 20:
            spike_lambda = 0
        else:
            spike_lambda = 1e-3
        spike_rate = total_spikes / (total_neu + 1e-9)
            
        # Use average output as final prediction
        y_mean = y_sum / T
        
        # Calculate loss
        loss = bce_weight * bce(y_mean, vox_gt) + (dice_weight * dice_loss(y_mean, vox_gt))

        # loss = loss + spike_lambda * spike_rate
        # loss = loss + spike_lambda * np.log(1 + spike_rate)
        loss = loss + spike_lambda * (spike_rate ** 2)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_iou = iou_score(y_mean.detach(), vox_gt.detach()).item()
        total_iou += batch_iou
        batch_count += 1
        
        # Log batch progress
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / batch_count
            avg_iou = total_iou / batch_count
            print(f"Epoch {epoch:03d}/{total_epochs} | Batch {batch_idx:03d} | Loss {avg_loss:.4f} | IoU {avg_iou:.3f} | Spike rate {spike_rate:.6f}")
    
    n = len(dataloader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def evaluate_snn(Smonitor,snn_model, dataloader, config, log_interval=5):
    """
    Evaluate SNN model performance
    
    Args:
        snn_model: SNN model
        dataloader: Data loader
        config: Config object
        log_interval: Number of batches between logging
        
    Returns:
        Average loss and average IoU
    """
    device = config.device
    dice_weight = config.dice_weight
    bce_weight = config.bce_weight
    snn_model.eval()
    
    total_loss = 0.0
    total_iou = 0.0
    bce = nn.BCEWithLogitsLoss()
    T = config.snn_steps
    batch_count = 0
    
    for batch_idx, (pts, vox_gt) in enumerate(dataloader, 1):
        pts = pts.to(device, non_blocking=True)  # [B, N, 3]
        vox_gt = vox_gt.to(device, non_blocking=True)  # [B, G, G, G]
        
        # Reset network state
        functional.reset_net(snn_model)
        
        # Run through multiple timesteps and accumulate outputs
        y_sum = 0
        total_spikes = 0
        total_neu = 0
        for t in range(T):
            y_t = snn_model(pts)  # [B, G, G, G]
            y_sum += y_t
            
        # Use average output as final prediction
        y_mean = y_sum / T

        for rec in Smonitor.records:
            total_spikes += (rec > 0).sum().item()
            total_neu += rec.numel()

        Smonitor.clear_recorded_data()

        spike_lambda = 1e-3
        spike_rate = total_spikes / (total_neu + 1e-9)
        
        # Calculate loss and IoU
        loss = bce_weight * bce(y_mean, vox_gt) + (dice_weight * dice_loss(y_mean, vox_gt))
        # loss = loss + spike_lambda * spike_rate
        # loss = loss + spike_lambda * np.log(1+spike_rate)
        loss = loss + spike_lambda * (spike_rate**2)
        total_loss += loss.item()
        total_iou += iou_score(y_mean, vox_gt).item()
        batch_count += 1
        
        # Log batch progress
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / batch_count
            avg_iou = total_iou / batch_count
            print(f"Val Batch {batch_idx:03d} | Loss {avg_loss:.4f} | IoU {avg_iou:.3f} | Spike rate {spike_rate:.6f}")
    
    n = len(dataloader)
    return total_loss / n, total_iou / n


if __name__ == "__main__":
    from model import AE
    from ann2snn import convert_ann_to_snn
    import torch
    
    config = Config()
    device = config.device
    
    # 1. Load original ANN model
    print("Loading ANN model...")
    ann_model = AE().to(device)
    ann_model.load_state_dict(torch.load("best_model_new.pth", map_location=device))
    
    # 2. Prepare conversion dataset (for ANN2SNN conversion)
    print("Preparing conversion dataset...")
    tr_dataset = PcdDataset(config=config, split="train")
    tr_dataloader = DataLoader(
        tr_dataset, 
        batch_size=config.batch_size_ann2snn, 
        shuffle=True, 
        collate_fn=collate_points
    )
    
    # 3. Prepare training and validation data
    print("Preparing training data...")
    train_dataset = PcdDataset(config=config, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_snn_train,
        shuffle=True,
        collate_fn=collate_points,
        num_workers=config.num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )
    
    val_dataset = PcdDataset(config=config, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_ann2snn_test,
        shuffle=False,
        collate_fn=collate_points,
        num_workers=config.num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True
    )

    modes = ['max', '99.9%', 1/2, 1/3, 1/4, 1/5]
    for mode in modes:
        # 3. Convert to SNN model
        # print(f"Converting ANN to SNN (mode={mode})...")
        snn_model = convert_ann_to_snn(ann_model, tr_dataloader, device, mode=mode)
        print(snn_model)
        snn_model = snn_model.to(device)

        Smonitor = monitor.OutputMonitor(snn_model, neuron.IFNode)
        # 5. Setup optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(snn_model.parameters(), lr=config.lr / 10)  # Use smaller lr for fine-tuning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        # 6. Fine-tuning training
        print("\nStarting SNN fine-tuning training...")
        best_iou = 0.0
        snn_epochs = 60  # Number of epochs for SNN fine-tuning

        for epoch in range(1, snn_epochs + 1):
            tr_loss, tr_iou = train_snn_epoch(Smonitor,snn_model, train_dataloader, optimizer, config, epoch=epoch, total_epochs=snn_epochs, log_interval=10)
            print(f"\n=> Epoch {epoch:03d}/{snn_epochs} | Train Loss {tr_loss:.4f} IoU {tr_iou:.3f}")

            val_loss, val_iou = evaluate_snn(Smonitor,snn_model, val_dataloader, config, log_interval=5)
            print(f"=> Epoch {epoch:03d}/{snn_epochs} | Val Loss {val_loss:.4f} IoU {val_iou:.3f}")

            scheduler.step()

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(snn_model.state_dict(), "MFT_best_snn_model_"+str(mode)+".pth")
                # torch.save(snn_model.encoder.state_dict(), f"best_encoder_snn_model_"+str(mode)+".pth")
                # torch.save(snn_model.decoder.state_dict(), f"best_decoder_snn_model_"+str(mode)+".pth")
                print(f"✓ Saved best SNN model to best_snn_model.pth (IoU: {val_iou:.4f})")

        # 7. Test set evaluation
        print("\nEvaluating on test set...")
        test_dataset = PcdDataset(config=config, split="test")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size_ann2snn_test,
            shuffle=False,
            collate_fn=collate_points,
            num_workers=config.num_workers,
            pin_memory=True if device == "cuda" else False,
            persistent_workers=True
        )

        test_loss, test_iou = evaluate_snn(Smonitor,snn_model, test_dataloader, config)
        print(f"Test results - Loss: {test_loss:.4f}, IoU: {test_iou:.4f}")
