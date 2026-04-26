import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss

class MultiLevelDetectionLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, lambda_=1):
        super(MultiLevelDetectionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_

    def _dict_to_list(self, targets):
        """딕셔너리 형태의 targets를 stride 순서대로 리스트화 (s8, s16, s32)"""
        ordered_targets = []
        for s in [8, 16, 32]:
            hm = targets[f'hm_s{s}']
            reg = targets[f'reg_s{s}']
            mask = targets[f'mask_s{s}']
            ordered_targets.append((hm, reg, mask))
        return ordered_targets

    def forward(self, predictions, targets_dict):
        targets = self._dict_to_list(targets_dict)
        total_cls_loss = 0
        total_reg_loss = 0
        
        for i in range(3):
            pred_hm, pred_reg = predictions[i]
            gt_hm, gt_reg, mask = targets[i]
            
            # 1. Focal Loss (CenterNet style)
            num_pos = mask.float().sum()
            num_pos = torch.clamp(num_pos, min=1.0)
            
            pred_hm = torch.clamp(pred_hm, min=1e-4, max=1-1e-4)
            
            # Gaussian 반경 내 포인트들에 대한 penalty 적용
            pos_mask = (gt_hm == 1.0).float()
            neg_mask = (gt_hm < 1.0).float()

            print(f"pred_hm shape: {pred_hm.shape}")
            print(f"pos_mask shape: {pos_mask.shape}")

            pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_mask
            neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * torch.pow(1 - gt_hm, self.beta) * neg_mask
            
            total_cls_loss -= (pos_loss.sum() + neg_loss.sum()) / num_pos # 현재 total_cls_loss를 num_pos로 나누고 있습니다. 하지만 3개의 레벨(S8, S16, S32) 전체에 대해 하나의 num_pos로 나누는 것이 아니라, 각 레벨별로 나누고 있습니다. 만약 특정 레벨에 객체가 하나도 없다면(num_pos=1), 해당 레벨의 neg_loss가 전체 로스를 지배할 수 있습니다. 권장: 3개 레벨의 모든 pos_mask 합계를 구한 뒤 마지막에 한 번 나누는 방식이 더 안정적입니다.

            # 2. Regression (GIoU)
            mask_bool = mask.squeeze(1) > 0.5 # [B, H, W]
            if mask_bool.any():
                # mask_bool이 True인 위치의 인덱스 추출 (B, Y, X)
                batch_idx, y_idx, x_idx = torch.where(mask_bool)

                # 예측값과 타겟값 추출
                p_regs = pred_reg.permute(0, 2, 3, 1)[mask_bool] # [N, 4] -> [w, h, ox, oy]
                t_regs = gt_reg.permute(0, 2, 3, 1)[mask_bool] # [N, 4] -> [w, h, ox, oy]

                # GIoU를 위한 x1, y1, x2, y2 좌표 변환
                # t_regs: [w, h, ox, oy]
                # 중심점 (cx, cy) = (grid_x + offset_x, grid_y + offset_y)

                def decode_to_bbox(regs, x_idx, y_idx):
                    # w, h는 항상 양수여야 하므로 모델 출력에 ReLU 등이 적용되어야 함
                    w, h = regs[:, 0], regs[:, 1]
                    ox, oy = regs[:, 2], regs[:, 3]

                    # 중심점 계산
                    cx, cy = x_idx.float() + ox, y_idx.float() + oy

                    # x1, y1, x2, y2로 변환
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    return torch.stack([x1, y1, x2, y2], dim=-1)

                p_boxes = decode_to_bbox(p_regs, x_idx, y_idx)
                t_boxes = decode_to_bbox(t_regs, x_idx, y_idx)
                
                reg_loss = generalized_box_iou_loss(p_boxes, t_boxes, reduction='mean')
                total_reg_loss += reg_loss


        print(f"Class loss : {total_cls_loss}, regression loss: {total_reg_loss}")

        return (total_cls_loss + self.lambda_ * total_reg_loss) / 3.0 # 주의: 학습 초기의 cls loss와 reg loss를 비교하면서 lambda_ 약간 조절하기



"""
import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss

class MultiLevelDetectionLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(MultiLevelDetectionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets):
        total_cls_loss = 0
        total_reg_loss = 0
        
        for i in range(3):
            pred_hm, pred_reg = predictions[i]
            gt_hm, gt_reg, mask = targets[i]
            
            # 1. Focal Loss 수정 (식 정교화)
            num_pos = mask.float().sum()
            # 0으로 나누기 방지
            num_pos = torch.clamp(num_pos, min=1.0)
            
            # Numerical Stability를 위해 clamp 사용
            pred_hm = torch.clamp(pred_hm, min=1e-4, max=1-1e-4)
            
            pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * gt_hm
            # (1 - gt_hm) 중복 제거
            neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * torch.pow(1 - gt_hm, self.beta)
            
            total_cls_loss -= (pos_loss.sum() + neg_loss.sum()) / num_pos

            # 2. Regression (GIoU) 수정
            mask_bool = mask.squeeze(1) > 0.5
            if mask_bool.any():
                # 중요: pred_reg에 적절한 activation(예: sigmoid)이 적용되어 있어야 함
                p_regs = pred_reg.permute(0, 2, 3, 1)[mask_bool]
                t_regs = gt_reg.permute(0, 2, 3, 1)[mask_bool]
                
                # 여기서 p_regs가 실제 x1, y1, x2, y2 좌표인지 반드시 확인!
                reg_loss = generalized_box_iou_loss(p_regs, t_regs, reduction='mean')
                total_reg_loss += reg_loss

        return total_cls_loss / 3.0 + total_reg_loss / 3.0
"""