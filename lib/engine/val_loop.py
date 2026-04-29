import wandb
import torch
from torch import amp
from lib.utils.utils import post_process

@torch.no_grad()
def validate_with_map(backbone, head, loader, device, metric, epoch):
    backbone.eval()
    head.eval()
    metric.reset()
    
    strides = [8, 16, 32]

    for imgs, targets in loader:
        imgs = imgs.to(device)
        
        p3, p4, p5 = backbone(imgs)
        preds = head(p3, p4, p5)
        
        pred_hms = [p[0] for p in preds]
        pred_regs = [p[1] for p in preds]

        # 예측값 변환 (Post-processing)
        batch_preds = []
        for b_idx in range(imgs.size(0)):
            single_hms = [hm[b_idx:b_idx+1] for hm in pred_hms]
            single_regs = [reg[b_idx:b_idx+1] for reg in pred_regs]
            
            # 히트맵에서 최종 박스 추출
            decoded_batch = post_process(single_hms, single_regs, strides, threshold=0.1)
            decoded = decoded_batch[0]
            
            if len(decoded) > 0:
                batch_preds.append({
                    "boxes": torch.stack([d['box'] for d in decoded]).to(device),
                    "scores": torch.stack([d['score'] for d in decoded]).to(device),
                    "labels": torch.stack([d['class_id'] for d in decoded]).to(device)
                })
            else:
                batch_preds.append({
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=device)
                })
                
        # 정답값 변환 (Dataset에서 이미 처리됨)
        batch_targets = []
        for t in targets: # 리스트 내부의 개별 타겟 딕셔너리에 접근
            batch_targets.append({
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device)
            })

        metric.update(batch_preds, batch_targets)

    results = metric.compute()

    wandb.log({
        "val/mAP": results["map"].item(),
        "val/mAP_50": results["map_50"].item(),
        "epoch": epoch
    })
    
    print(f"[{epoch}] Validation mAP: {results['map']:.4f}")
    return results