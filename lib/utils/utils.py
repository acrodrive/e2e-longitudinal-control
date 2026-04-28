import os
import torch
import torch.nn.functional as F

def decode_to_bbox(regs, x_idx, y_idx):
    """
    모델의 출력값을 상응하는 좌표계의 [x1, y1, x2, y2]로 변환합니다.
    """

    w = torch.clamp(regs[:, 0], min=1e-3) 
    h = torch.clamp(regs[:, 1], min=1e-3)
    
    ox, oy = regs[:, 2], regs[:, 3]
    cx, cy = x_idx.float() + ox, y_idx.float() + oy
    
    x1 = (cx - w / 2)
    y1 = (cy - h / 2)
    x2 = (cx + w / 2)
    y2 = (cy + h / 2)
    
    return torch.stack([x1, y1, x2, y2], dim=-1)

def decode_to_bbox_in_raw(regs, x_idx, y_idx, stride):
    # regs: [N, 4] -> (w, h, ox, oy)
    # 모델이 음수를 뱉더라도 최소 0.1 픽셀 이상의 크기를 갖도록 클램핑
    w = torch.clamp(regs[:, 0], min=1e-3) 
    h = torch.clamp(regs[:, 1], min=1e-3)
    
    ox, oy = regs[:, 2], regs[:, 3]
    cx, cy = x_idx.float() + ox, y_idx.float() + oy
    
    x1 = (cx - w / 2) * stride
    y1 = (cy - h / 2) * stride
    x2 = (cx + w / 2) * stride
    y2 = (cy + h / 2) * stride
    
    return torch.stack([x1, y1, x2, y2], dim=-1)

"""def decode_to_bbox_in_raw(regs, x_idx, y_idx, stride):
    #모델의 피처맵 기준 출력값을 원본 영상 좌표계의 [x1, y1, x2, y2]로 변환합니다.

    w, h = regs[:, 0], regs[:, 1]
    ox, oy = regs[:, 2], regs[:, 3]
    cx, cy = x_idx.float() + ox, y_idx.float() + oy
    
    x1 = (cx - w / 2) * stride
    y1 = (cy - h / 2) * stride
    x2 = (cx + w / 2) * stride
    y2 = (cy + h / 2) * stride
    
    return torch.stack([x1, y1, x2, y2], dim=-1)"""

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

def get_local_maximum(heat, kernel=3):
    """히트맵에서 3x3 Max pooling을 통해 로컬 최대값(피크)만 남깁니다."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def post_process(pred_hms, pred_regs, strides, threshold=0.3, top_k=100):
    """
    모델의 출력(Multi-level)을 해석하여 최종 [x1, y1, x2, y2, score, cls] 리스트를 반환합니다.
    """
    detections = []
    
    for i in range(len(pred_hms)):
        hm = get_local_maximum(pred_hms[i]) # [1, C, H, W]
        reg = pred_regs[i]                  # [1, 4, H, W]
        stride = strides[i]
        
        batch, cat, height, width = hm.size()
        
        # 스코어와 인덱스 추출
        scores, inds = torch.topk(hm.view(batch, -1), top_k)
        
        classes = (inds // (height * width)).int()
        inds = inds % (height * width)
        ys = (inds // width).int()
        xs = (inds % width).int()
        
        # 레그레션 값 추출 (w, h, ox, oy)
        reg = reg.permute(0, 2, 3, 1).contiguous().view(batch, -1, 4)
        reg = reg.gather(1, inds.unsqueeze(-1).repeat(1, 1, 4)) # [B, top_k, 4]
        
        scores = scores.cpu().numpy()[0]
        classes = classes.cpu().numpy()[0]
        xs = xs.cpu().numpy()[0]
        ys = ys.cpu().numpy()[0]
        reg = reg.cpu().numpy()[0]

        for k in range(len(scores)):
            if scores[k] < threshold:
                continue
            
            w, h, ox, oy = reg[k]
            # 그리드 좌표 + 오프셋
            cx, cy = (xs[k] + ox) * stride, (ys[k] + oy) * stride
            # 실제 픽셀 크기로 변환
            w, h = w * stride, h * stride
            
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'score': float(scores[k]),
                'class_id': int(classes[k])
            })
            
    return detections