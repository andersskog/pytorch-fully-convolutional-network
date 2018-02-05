import numpy as np
import torch

def accuracy(batch_data, pred):
    (imgs, segs, infos) = batch_data
    _, preds = torch.max(pred.data.cpu(), dim=1)
    valid = (segs >= 0)
    acc = 1.0 * torch.sum(valid * (preds == segs)) / (torch.sum(valid) + 1e-10)
    return acc, torch.sum(valid)