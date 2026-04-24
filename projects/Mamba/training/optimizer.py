import torch.optim as optim

def get_optimizer(model, config):
    # 백본은 보통 더 낮은 학습률을 적용하는 것이 관례입니다.
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr * 0.1, # 백본은 10배 작게
        },
    ]
    
    optimizer = optim.AdamW(param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    
    # StepLR 또는 CosineAnnealing 활용
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    
    return optimizer, scheduler