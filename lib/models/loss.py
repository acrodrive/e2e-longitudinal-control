import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss
from lib.utils.utils import decode_to_bbox

class MultiLevelDetectionLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, lambda_=2, threshold=0.5):
        super(MultiLevelDetectionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_

    def forward(self, pred_hms, pred_boxes, gt_hms, gt_boxes, masks):
        # targets = self._dict_to_list(targets_dict)
        total_cls_loss, total_reg_loss, total_num_pos = 0, 0, 0
        
        for i in range(3):
            # pred_hm, pred_reg = predictions[i]
            # gt_hm, gt_reg, masks = targets[i]
            # 
            # # 1. Classification (Focal Loss)
            mask = masks[i]
            num_pos = mask.float().sum()
            total_num_pos += num_pos
            
            # pred_hm = torch.clamp(pred_hm, min=1e-4, max=1-1e-4)
            pred_hm = torch.clamp(pred_hms[i], min=1e-7, max=1-1e-7)
            
            pos_masks = (gt_hms[i] == 1.0).float()
            neg_masks = (gt_hms[i] < 1.0).float()

            pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_masks
            neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * torch.pow(1 - gt_hms[i], self.beta) * neg_masks

            # 누적할 때 마이너스를 붙여서 양수로 저장
            total_cls_loss += -(pos_loss.sum() + neg_loss.sum())
            
            # 2. Regression (GIoU)
            # masks_bool = masks.squeeze(1) > 0.5
            # if masks_bool.any():
            #     batch_idx, y_idx, x_idx = torch.where(masks_bool)
            #     p_regs = pred_reg.permute(0, 2, 3, 1)[masks_bool]
            #     t_regs = gt_reg.permute(0, 2, 3, 1)[masks_bool]
            #     p_boxes = decode_to_bbox(p_regs, x_idx, y_idx)
            #     t_boxes = decode_to_bbox(t_regs, x_idx, y_idx)
            if pred_boxes[i].numel() > 0:
                total_reg_loss += generalized_box_iou_loss(
                    pred_boxes[i], 
                    gt_boxes[i], 
                    reduction='sum' ########################################## mean 방식도 고려하기
                )
            # total_reg_loss += generalized_box_iou_loss(pred_boxes[i], gt_boxes[i], reduction='sum')
                
        total_num_pos = torch.clamp(total_num_pos, min=1.0)
        cls_loss = total_cls_loss / total_num_pos
        reg_loss = total_reg_loss / total_num_pos

        # print(f"Loss(Total): {cls_loss.item() + self.lambda_ * reg_loss.item():.4f})        
        return (total_cls_loss / total_num_pos) + (self.lambda_ * total_reg_loss / total_num_pos)