import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    """
    Transformer Decoder의 출력(Hidden States)을 받아 
    클래스 분류와 바운딩 박스 회귀를 수행하는 헤드.
    """
    def __init__(self, d_model=256, num_classes=80):
        super().__init__()
        
        # 1. Classification Head (배경 클래스 포함: num_classes + 1)
        self.class_head = nn.Linear(d_model, num_classes + 1)
        
        # 2. Bounding Box Regression Head
        # [cx, cy, w, h] 4개의 값을 예측
        # 성능 향상을 원한다면 단순 Linear 대신 MLP(Multi-Layer Perceptron)로 확장 가능
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)
        )

    def forward(self, hs):
        """
        Args:
            hs: Decoder에서 나온 Hidden States [batch, num_queries, d_model]
        Returns:
            dict: {
                "pred_logits": 클래스 예측값 [batch, num_queries, num_classes + 1],
                "pred_boxes": 박스 좌표 예측값 [batch, num_queries, 4]
            }
        """
        outputs_class = self.class_head(hs)
        outputs_coord = self.bbox_head(hs).sigmoid() # 좌표는 0~1 사이 값으로 정규화하여 출력
        
        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}