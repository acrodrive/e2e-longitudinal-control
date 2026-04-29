import wandb
import torch
from torch import amp
from lib.utils.utils import post_process
"""
@torch.no_grad()
def validate_one_epoch(backbone, head, loader, criterion, device, metrics, epoch):
    backbone.eval()
    head.eval()
    val_loss = 0.0
    metrics.reset()

    for i, batch in enumerate(loader):
        if batch is None: continue
        
        imgs, targets_dict = batch
        imgs = imgs.to(device)
        targets_dict = {k: v.to(device) for k, v in targets_dict.items()}
        
        # Inference (Mixed Precision)
        with amp.autocast(device_type=device.type) if device.type == 'cuda' else torch.enable_grad():
            p3, p4, p5 = backbone(imgs)
            preds = head(p3, p4, p5)

            collected_data = {
                'pred_hms': [], 'pred_boxes': [], 
                'gt_hms': [], 'gt_boxes': [], 'masks': []
            }

            for j in range(3):
                pred_hm, pred_reg = preds[j]
                gt_hm, gt_reg, mask = targets_dict[f'hm_s{8*2**j}'], targets_dict[f'reg_s{8*2**j}'], targets_dict[f'mask_s{8*2**j}']

                from lib.utils.utils import decode_to_bbox
                mask_bool = mask.squeeze(1) > 0.5
                if mask_bool.any():
                    batch_idx, y_idx, x_idx = torch.where(mask_bool)
                    pred_regs = pred_reg.permute(0, 2, 3, 1)[mask_bool]
                    gt_regs = gt_reg.permute(0, 2, 3, 1)[mask_bool]
                    pred_box = decode_to_bbox(pred_regs, x_idx, y_idx)
                    gt_box = decode_to_bbox(gt_regs, x_idx, y_idx)
                else:
                    pred_box = torch.empty((0, 4), device=device)
                    gt_box = torch.empty((0, 4), device=device)

                collected_data['pred_hms'].append(pred_hm)
                collected_data['gt_hms'].append(gt_hm)
                collected_data['pred_boxes'].append(pred_box)
                collected_data['gt_boxes'].append(gt_box)
                collected_data['masks'].append(mask)
            
            tot_loss, _, _ = criterion(**collected_data)
        
        val_loss += tot_loss.item()
        
        # Metrics Update
        for j in range(3):
            metrics.update_conf(collected_data['pred_hms'][j], (collected_data['gt_hms'][j] == 1.0).float())
            if collected_data['pred_boxes'][j].numel() > 0:
                metrics.update_mae(collected_data['pred_boxes'][j], collected_data['gt_boxes'][j], stride=8 * (2**j))

    avg_val_loss = val_loss / len(loader)
    stats = metrics.compute()
    
    return avg_val_loss, stats
"""

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
            decoded = post_process(single_hms, single_regs, strides, threshold=0.1)
            
            batch_preds.append({
                "boxes": torch.tensor([d['box'] for d in decoded]).to(device),
                "scores": torch.tensor([d['score'] for d in decoded]).to(device),
                "labels": torch.tensor([d['class_id'] for d in decoded]).to(device)
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