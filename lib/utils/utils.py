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

    filename = os.path.join(folder, f"checkpoint_epoch_{epoch}.pth.tar")
    torch.save(state, filename)
    
    last_path = os.path.join(folder, "last_model.pth.tar")
    torch.save(state, last_path)
    print(f"--- Checkpoint saved: {filename} ---")

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    if len(batch) == 0:
        return None
    
    return torch.utils.data.dataloader.default_collate(batch)

def collate_fn_for_validation(batch):
    # None 샘플 제거
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        return None

    # 이미지(imgs)와 타겟(targets)을 분리
    # batch는 [(img1, target1), (img2, target2), ...] 구조임
    imgs, targets = zip(*batch)
    
    # 이미지는 모두 크기가 동일하므로(Resize 처리됨) 텐서로 묶음
    imgs = torch.stack(imgs, dim=0)
    
    # 타겟은 객체 수가 제각각이므로 묶지 않고 '리스트' 그대로 반환
    # targets는 (target1, target2, ...) 구조의 튜플/리스트가 됨
    return imgs, targets

def get_local_maximum(heat, kernel=3):
    """히트맵에서 3x3 Max pooling을 통해 로컬 최대값(피크)만 남깁니다."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def post_process(pred_hms, pred_regs, strides, threshold=0.3, top_k=100):
    """
    모든 연산을 GPU에서 수행하여 CPU-GPU 전송 오버헤드를 제거합니다.
    """
    batch_size = pred_hms[0].shape[0]
    device = pred_hms[0].device
    all_detections = [[] for _ in range(batch_size)]
    
    for i in range(len(pred_hms)):
        # 1. Local Maximum 추출 (이미 구현된 함수 사용)[cite: 8]
        hm = get_local_maximum(pred_hms[i]) 
        reg = pred_regs[i]
        stride = strides[i]
        
        _, C, H, W = hm.shape
        
        # 2. 상위 K개 후보 추출 (Batch 단위 처리)[cite: 8]
        # [B, C*H*W]
        scores, inds = torch.topk(hm.view(batch_size, -1), top_k)
        
        classes = (inds // (H * W)).int()
        inds = inds % (H * W)
        ys = (inds // W).float()
        xs = (inds % W).float()
        
        # 3. Regression 값 일괄 추출[cite: 8]
        # [B, H*W, 4]
        reg = reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        # [B, top_k, 4]
        reg_candidates = torch.gather(reg, 1, inds.unsqueeze(-1).expand(-1, -1, 4))
        
        # 4. 벡터화된 좌표 디코딩[cite: 8]
        # reg_candidates: (w, h, ox, oy)
        w, h = reg_candidates[..., 0] * stride, reg_candidates[..., 1] * stride
        cx = (xs + reg_candidates[..., 2]) * stride
        cy = (ys + reg_candidates[..., 3]) * stride
        
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        
        # 5. 임계값 필터링 및 결과 정리
        for b in range(batch_size):
            keep = scores[b] >= threshold
            if not keep.any():
                continue
                
            # 최종 박스 구성 [N, 4]
            batch_boxes = torch.stack([x1[b, keep], y1[b, keep], x2[b, keep], y2[b, keep]], dim=-1)
            batch_scores = scores[b, keep]
            batch_labels = classes[b, keep]
            
            # 결과 저장 (Dictionary 형태 유지)[cite: 3]
            for k in range(len(batch_scores)):
                all_detections[b].append({
                    'box': batch_boxes[k],
                    'score': batch_scores[k],
                    'class_id': batch_labels[k]
                })
                
    return all_detections