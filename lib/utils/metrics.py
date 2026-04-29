import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class MetricsCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        # Confidence 관련
        self.total_conf = 0.0
        self.conf_count = 0
        # MAE 관련
        self.total_mae = 0.0
        self.reg_count = 0

    def update_conf(self, pred_hm, pos_mask):
        """
        정답 중심점(pos_mask == 1) 위치에서 모델의 평균 예측 확률을 계산
        """
        if pos_mask.sum() > 0:
            # 정답인 픽셀들만 추출하여 평균 계산
            avg_conf = pred_hm[pos_mask == 1].mean().item()
            self.total_conf += avg_conf
            self.conf_count += 1

    def update_mae(self, p_boxes, t_boxes, stride):
        """MAE 계산 시 stride를 곱해 실제 픽셀 오차로 변환"""
        if p_boxes.numel() == 0:
            return
        
        # [N, 4] 형태에서 각 성분의 차이를 구하고 stride를 곱함
        pixel_error = torch.abs(p_boxes - t_boxes) * stride
        self.total_mae += pixel_error.sum().item()
        self.reg_count += p_boxes.numel()

    def compute(self):
        """최종 지표 계산"""
        avg_pos_conf = self.total_conf / (self.conf_count + 1e-6)
        avg_pixel_mae = self.total_mae / (self.reg_count + 1e-6)
        
        return {
            "avg_pos_conf": avg_pos_conf,
            "avg_pixel_mae": avg_pixel_mae
        }

class MAPCalculator:
    def __init__(self, device='cpu'):
        # box_format='xyxy'는 [x1, y1, x2, y2] 형식을 의미합니다.
        self.metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)

    def update(self, preds, targets):
        """
        preds: 리스트 [ {"boxes": tensor, "scores": tensor, "labels": tensor}, ... ]
        targets: 리스트 [ {"boxes": tensor, "labels": tensor}, ... ]
        """
        self.metric.update(preds, targets)

    def compute(self):
        """최종 mAP 지표 계산"""
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
        
"""import torch

class MetricsCalculator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_mae = 0
        self.reg_count = 0

    def update_f1(self, pred_hm, pos_mask):
        pred_binary = (pred_hm > self.threshold).float()
        self.total_tp += ((pred_binary == 1) & (pos_mask == 1)).sum().item()
        self.total_fp += ((pred_binary == 1) & (pos_mask == 0)).sum().item()
        self.total_fn += ((pred_binary == 0) & (pos_mask == 1)).sum().item()

    def update_mae(self, p_boxes, t_boxes, stride):
        pixel_error = torch.abs(p_boxes - t_boxes) * stride
        self.total_mae += pixel_error.sum().item()
        
        self.reg_count += p_boxes.size(0)

    def compute(self):
        precision = self.total_tp / (self.total_tp + self.total_fp + 1e-6)
        recall = self.total_tp / (self.total_tp + self.total_fn + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        mae = self.total_mae / (self.reg_count + 1e-6)
        
        return {
            "f1_score": f1_score,
            "mae": mae
        }"""