import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random


def compute_grad_vector(model, loss_fn, x, y):
    device = next(model.parameters()).device  # detect model's device
    x = x.view(x.shape[0], -1).to(device)
    y = y.to(device)
    model.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    grad_vec = torch.cat(grads)
    return grad_vec.detach()


def measure_gradient_confusion(model, loss_fn, dataloader, num_pairs=100):
    device = next(model.parameters()).device  # detect model's device
    similarities = []
    data_iter = list(dataloader)
    for _ in range(num_pairs):
        batch1, batch2 = random.sample(data_iter, 2)
        x1, y1 = batch1[0].to(device), batch1[1].to(device)
        x2, y2 = batch2[0].to(device), batch2[1].to(device)

        g1 = compute_grad_vector(model, loss_fn, x1, y1)
        g2 = compute_grad_vector(model, loss_fn, x2, y2)

        cos_sim = torch.nn.functional.cosine_similarity(
            g1.unsqueeze(0), g2.unsqueeze(0)
        )
        similarities.append(cos_sim.item())

    return similarities
