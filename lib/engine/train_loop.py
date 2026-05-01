import wandb
import torch
from torch import amp
from lib.utils.utils import decode_reg_to_bbox, decode_to_bbox_in_raw

def train_one_epoch(backbone, head, loader, criterion, optimizer, scheduler, scaler, device, metrics, epoch, epochs):
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
        
        context = amp.autocast(device_type=device.type) if is_cuda else torch.enable_grad() #contextlib.nullcontext()

        with context:
            p3, p4, p5 = backbone(imgs)
            preds = head(p3, p4, p5)

            collected_data = {
            'pred_hms': [], 'pred_regs': [], 
            'gt_hms': [], 'gt_regs': [], 'masks': []
            }

            for j in range(3):
                # GT data
                gt_hm, gt_reg, mask = targets_dict[f'hm_s{8*2**j}'], targets_dict[f'reg_s{8*2**j}'], targets_dict[f'mask_s{8*2**j}']
                
                # Model Output
                pred_hm, pred_reg = preds[j]

                mask_bool = mask.squeeze(1) > 0.5
                """
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
                """
                if mask_bool.any():
                    active_pred_reg = pred_reg.permute(0, 2, 3, 1)[mask_bool]
                    active_gt_reg = gt_reg.permute(0, 2, 3, 1)[mask_bool]
                    metrics.update_mae(active_pred_reg, active_gt_reg, 8*(2**j)) # TODO 이 함수 점검

                collected_data['pred_hms'].append(pred_hm)
                collected_data['gt_hms'].append(gt_hm)
                collected_data['pred_regs'].append(pred_reg)
                collected_data['gt_regs'].append(gt_reg)
                collected_data['masks'].append(mask)
                
                pos_mask = (gt_hm == 1.0).float()
                metrics.update_conf(pred_hm, pos_mask)
            
                """
                if pred_box.numel() > 0:
                    metrics.update_mae(pred_box, gt_box, stride=8 * (2**j))
                """

            tot_loss, cls_loss, reg_loss, iou_loss, offset_loss = criterion(**collected_data)
        
        if is_cuda:
            scaler.scale(tot_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            tot_loss.backward()
            optimizer.step()
        
        epoch_loss += tot_loss.item()

        if i % 10 == 0:
            wandb.log({
                "batch_loss": tot_loss.item(),
                "class_loss": cls_loss.item(),
                "reg_loss": reg_loss.item(),
                "iou_loss": iou_loss.item(),
                "offset_loss": offset_loss.item(),
                "global_step": epoch * len(loader) + i
            })
        if i % 100 == 0 and i != 0:
            print(f"[Epoch {epoch}/{epochs}] Batch: {i}/{len(loader)}")
            """
            wandb.log({
                "train/heatmap_sample": [wandb.Image(collected_data['pred_hms'][0][0, 0].detach().cpu().numpy())],
                "train/gt_heatmap_sample": [wandb.Image(collected_data['gt_hms'][0][0, 0].detach().cpu().numpy())]
            })
            """

    avg_loss = epoch_loss / len(loader)
    stats = metrics.compute()
    
    wandb.log({
        "epoch": epoch,
        "train_avg_loss": avg_loss,
        "confidence": stats['avg_pos_conf'],
        "mae": stats['avg_pixel_mae']
    })

    if scheduler is not None:
        scheduler.step()
    
    return avg_loss