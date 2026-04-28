import wandb
import torch
from torch import amp
from lib.utils.utils import decode_to_bbox, decode_to_bbox_in_raw
import lib.utils.metrics

def train_one_epoch(backbone, head, loader, criterion, optimizer, scaler, device, metrics, epoch, epochs):
    backbone.train()
    head.train()
    epoch_loss = 0.0
    is_cuda = device.type == 'cuda'
    metrics.reset()

    for i, batch in enumerate(loader):
        if batch is None: continue
        
        imgs, targets_dict = batch
        imgs = imgs.to(device)
        targets_dict = {k: v.to(device) for k, v in targets_dict.items()}
        
        optimizer.zero_grad()
        
        # Mixed Precision 지원
        context = amp.autocast(device_type=device.type) if is_cuda else torch.enable_grad()

        with context:
            p3, p4, p5 = backbone(imgs)
            preds = head(p3, p4, p5)

            collected_data = {
            'pred_hms': [], 'pred_boxes': [], 
            'gt_hms': [], 'gt_boxes': [], 'masks': []
            }

            for j in range(3):
                pred_hm, pred_reg = preds[j]
                gt_hm, gt_reg, mask = targets_dict[f'hm_s{8*2**j}'], targets_dict[f'reg_s{8*2**j}'], targets_dict[f'mask_s{8*2**j}']

                pred_hm = torch.clamp(pred_hm, min=1e-4, max=1-1e-4)

                mask_bool = mask.squeeze(1) > 0.5
                if mask_bool.any():
                    batch_idx, y_idx, x_idx = torch.where(mask_bool)
                    pred_regs = pred_reg.permute(0, 2, 3, 1)[mask_bool]
                    gt_regs = gt_reg.permute(0, 2, 3, 1)[mask_bool]

                    pred_box = decode_to_bbox(pred_regs, x_idx, y_idx)
                    gt_box = decode_to_bbox(gt_regs, x_idx, y_idx)
                else:
                    # 객체가 없는 경우 빈 텐서 전달 (Loss/Metrics에서 처리 가능하게)
                    pred_box = torch.empty((0, 4), device=device)
                    gt_box = torch.empty((0, 4), device=device)

                collected_data['pred_hms'].append(pred_hm)
                collected_data['gt_hms'].append(gt_hm)
                collected_data['pred_boxes'].append(pred_box)
                collected_data['gt_boxes'].append(gt_box)
                collected_data['masks'].append(mask)
            
            tot_loss, cls_loss, reg_loss = criterion(**collected_data)
        
        if is_cuda:
            scaler.scale(tot_loss).backward()
            #torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            tot_loss.backward()
            optimizer.step()
            
        epoch_loss += tot_loss.item()
        
        for j in range(3):
            # collected_data 딕셔너리에서 데이터를 가져와야 합니다.
            p_hm = collected_data['pred_hms'][j]
            g_hm = collected_data['gt_hms'][j]
            p_box = collected_data['pred_boxes'][j]
            g_box = collected_data['gt_boxes'][j]

            # [Confidence 업데이트] 정답 위치의 활성화 정도
            # gt_hms[j] == 1.0인 지점(Center)만 mask로 사용
            pos_mask = (g_hm == 1.0).float()
            metrics.update_conf(p_hm, pos_mask)
        
            # [MAE 업데이트] stride 반영
            if p_box.numel() > 0:
                metrics.update_mae(p_box, g_box, stride=8 * (2**j))

        #if i % 10 == 0 and i != 0:
        #    print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(loader)}] Loss: {loss.item():.4f}")

        if i % 10 == 0:
            wandb.log({
                "class_loss": cls_loss.item(),
                "reg_loss": reg_loss.item(),
                "batch_loss": tot_loss.item(),
                "global_step": epoch * len(loader) + i
            })
        if i % 100 == 0 and i != 0:
            print(f"train process: {i}")

    # 에폭이 끝나고 평균값 기록
    avg_loss = epoch_loss / len(loader)
    stats = metrics.compute() # 분리한 metrics 클래스 활용
    
    wandb.log({
        "epoch": epoch,
        "train_avg_loss": avg_loss,
        "f1_score": stats['avg_pos_conf'],
        "mae": stats['avg_pixel_mae']
    })
            
    return avg_loss, stats