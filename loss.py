import torch


def dice_loss(logits, targets):
    """
    Dice loss that accepts logits (will compute sigmoid internally).
    
    Args:
        logits: [B, G, G, G] - model output logits
        targets: [B, G, G, G] - ground truth binary voxels
    """
    # Convert logits to probabilities using sigmoid
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=[1,2,3])
    union = probs.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3])
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()

def iou_score(logits, targets, thresh=0.5, eps=1e-6):
    """
    IoU score that accepts logits (will compute sigmoid internally).
    
    Args:
        logits: [B, G, G, G] - model output logits
        targets: [B, G, G, G] - ground truth binary voxels
        thresh: threshold for binarization (applied after sigmoid)
    """
    # Convert logits to probabilities using sigmoid, then binarize
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    inter = (preds * targets).sum(dim=[1,2,3])
    union = (preds + targets - preds*targets).sum(dim=[1,2,3])
    iou = (inter + eps) / (union + eps)
    return iou.mean()