import os
import torch

def decode_to_bbox(regs, x_idx, y_idx):
    """
    모델의 출력값(regs)을 실제 좌표계의 [x1, y1, x2, y2]로 변환합니다.
    """
    w, h = regs[:, 0], regs[:, 1]
    ox, oy = regs[:, 2], regs[:, 3]
    cx, cy = x_idx.float() + ox, y_idx.float() + oy
    
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    return torch.stack([x1, y1, x2, y2], dim=-1)

def load_model_weights(backbone, head, optimizer, scaler, checkpoint_path):
    start_epoch = 0
    if not checkpoint_path: 
        print("=> checkpoint : None")
        return start_epoch
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Scaler 상태도 복구하여 학습 연속성 유지
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"=> Resumed from epoch {start_epoch}")
    else:
        print("=> No checkpoint found. Starting from pre-trained backbone.")
    return start_epoch

def save_checkpoint(state, epoch, folder="checkpoints"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # TODO: Include architecture name and timestamp in the filename.
    filename = os.path.join(folder, f"checkpoint_epoch_{epoch}.pth.tar")
    torch.save(state, filename)
    
    last_path = os.path.join(folder, "last_model.pth.tar")
    torch.save(state, last_path)
    print(f"--- Checkpoint saved: {filename} ---")

def collate_fn(batch): #참고: collate_fn에서 가변 길이의 박스 리스트를 처리할 수 있도록 수정이 필요하다고 합니다.
    batch = list(filter(lambda x: x is not None, batch))

    if len(batch) == 0:
        return None
    
    return torch.utils.data.dataloader.default_collate(batch)