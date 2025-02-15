import torch

from torch import nn

softmax = nn.Softmax(dim=-1)

def mask_from_pred(batch_pred : torch.Tensor) -> torch.Tensor:
    """
    Computes semantic masks for batch.
    Returns torch.Tensor of dtype=long
    batch : torch.Tensor - predicted value over batch of shape (batch_size, height, width, channels)
    """
    with torch.no_grad():
        batch_softmax = softmax(batch_pred)
    mask = torch.argmax(batch_softmax, dim=-1)
    return mask
