import torch

def compute_metrics(pred, target):
    mse = torch.mean((pred - target)**2)
    mae = torch.mean(torch.abs(pred - target))
    return {"MSE": mse.item(), "MAE": mae.item()}