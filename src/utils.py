import torch

from torch import nn

softmax = nn.Softmax(dim=-1)

def mask_from_pred(batch_pred : torch.Tensor, confidence : float = 0.5, one_channel : bool = True) -> torch.Tensor:
    """
    Computes semantic masks for batch.
    Returns torch.Tensor of dtype=long
    batch : torch.Tensor - predicted value over batch of shape (batch_size, height, width, channels)
    confidence : float - parameter to compute threshold function of softmax
    one_channel : bool - if True will return tensor of shape (batch_size, height, width) with values of
        [0, ..., num_classes-1] else will return shape (batch_size, height, width, num_classes) with {0, 1}
    """
    with torch.no_grad():
        batch_softmax = softmax(batch_pred)
    semantic_segmented = (batch_softmax > confidence).long()
    weighted = []
    for i in range(semantic_segmented.shape[-1]):
        if one_channel:
            weighted.append(i * semantic_segmented[:,:,:,i])
        else:
            weighted.append(semantic_segmented[:,:,:,i])
    mask = torch.stack(weighted, dim=-1).to(semantic_segmented.device).long()
    if one_channel:
        mask = mask.sum(-1)
    return mask
