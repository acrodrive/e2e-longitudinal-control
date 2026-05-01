import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss
from lib.utils.utils import decode_to_bbox

class MultiLevelDetectionLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, lambda_=5):
        super(MultiLevelDetectionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_

    def forward(self, pred_hms, pred_boxes, gt_hms, gt_boxes, masks):
        # targets = self._dict_to_list(targets_dict)
        total_cls_loss, total_reg_loss, total_num_pos = 0, 0, 0
        
        for i in range(3):
            # Positive sample 개수
            mask = masks[i]
            num_pos = mask.float().sum()
            total_num_pos += num_pos
            
            # Classification (Focal Loss)
            pred_hm = torch.clamp(pred_hms[i], min=1e-6, max=1-1e-6)
            
            pos_masks = (gt_hms[i] == 1.0).float()
            neg_masks = (gt_hms[i] < 1.0).float()

            pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_masks
            neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * torch.pow(1 - gt_hms[i], self.beta) * neg_masks

            # 누적할 때 마이너스를 붙여서 양수로 저장
            total_cls_loss += -(pos_loss.sum() + neg_loss.sum())
            
            # 2. Regression (CIoU)
            if pred_boxes[i].numel() > 0:
                reg_loss_sum = complete_box_iou_loss(
                    pred_boxes[i], 
                    gt_boxes[i], 
                    reduction='sum'
                )
                total_reg_loss += reg_loss_sum
                
        total_num_pos = torch.clamp(total_num_pos, min=1.0)
        
        cls_loss = total_cls_loss / total_num_pos
        reg_loss = total_reg_loss / total_num_pos
        
        tot_loss = cls_loss + (self.lambda_ * reg_loss)
        
        return tot_loss, cls_loss, reg_loss