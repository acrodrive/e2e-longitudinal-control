import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self, num_classes, matcher):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        # 배경(No-object) 클래스에 대한 가중치 조절이 필요할 수 있음
        self.empty_weight = 0.1 

    def forward(self, outputs, targets):
        # 1. 매칭 수행
        indices = self.matcher(outputs, targets)

        # 2. Classification Loss (Cross Entropy)
        # 모든 쿼리를 '배경'으로 초기화 후 매칭된 곳만 정답 라벨 부여
        src_logits = outputs["pred_logits"]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        # 매칭된 인덱스에 실제 라벨 채우기 로직 생략(복잡하므로 핵심 개념만 전달)
        # ... 

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

        # 3. Bbox L1 Loss
        # 매칭된 쿼리에 대해서만 L1 distance 계산
        # ...
        loss_bbox = F.l1_loss(matched_out_bbox, matched_tgt_bbox, reduction='sum')

        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox}