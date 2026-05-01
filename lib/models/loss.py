import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss
from lib.utils.utils import decode_reg_to_bbox
import torch.nn.functional as F

class MultiLevelDetectionLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, lambda_=5):
        super(MultiLevelDetectionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_

    def forward(self, pred_hms, pred_regs, gt_hms, gt_regs, masks):
        # targets = self._dict_to_list(targets_dict)
        total_cls_loss, total_reg_loss, total_num_pos = 0, 0, 0
        
        for i in range(3):
            # Positive sample 개수
            mask = masks[i]
            num_pos = mask.float().sum()
            total_num_pos += num_pos
            
            # Classification (Focal Loss)
            pred_hm = pred_hms[i]
            pred_hm = torch.clamp(pred_hm, min=1e-6, max=1-1e-6)
            
            pos_masks = (pred_hm == 1.0).float()
            neg_masks = (pred_hm < 1.0).float()

            pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_masks
            neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * torch.pow(1 - gt_hms[i], self.beta) * neg_masks

            # 누적할 때 마이너스를 붙여서 양수로 저장
            total_cls_loss += -(pos_loss.sum() + neg_loss.sum())
            
            # --- 2. Regression (CIoU + Offset L1) ---
            if num_pos > 0:
                # Mask를 이용해 객체가 있는 index 추출 [N]
                # mask shape [B, 1, H, W] -> B, H, W 순서로 flatten 하여 추출
                batch_idx, _, yy, xx = torch.where(mask > 0)
                
                # 예측값과 정답값에서 Positive 샘플만 추출 (N, 4)
                p_reg = pred_regs[i][batch_idx, :, yy, xx] 
                g_reg = gt_regs[i][batch_idx, :, yy, xx]

                # (A) CIoU Loss
                iou_loss = complete_box_iou_loss(
                    decode_reg_to_bbox(p_reg),
                    decode_reg_to_bbox(g_reg),
                    reduction='sum'
                )
                
                # (B) Offset Loss (L1): ox, oy 정보 정밀 학습 (p_reg[:, 2:], g_reg[:, 2:])
                offset_loss = F.l1_loss(p_reg[:, 2:], g_reg[:, 2:], reduction='sum')
                
                total_reg_loss += (iou_loss + offset_loss)
                
        total_num_pos = torch.clamp(total_num_pos, min=1.0)
        
        cls_loss = total_cls_loss / total_num_pos
        reg_loss = total_reg_loss / total_num_pos
        
        tot_loss = cls_loss + (self.lambda_ * reg_loss)
        
        return tot_loss, cls_loss, reg_loss, iou_loss / total_num_pos, offset_loss / total_num_pos