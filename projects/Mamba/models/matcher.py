import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    예측된 종단점(logits, boxes)과 실제 정답 사이의 최적 매칭을 찾습니다.
    """
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: {"pred_logits": [B, Q, C], "pred_boxes": [B, Q, 4]}
        targets: list of dicts [{"labels": [...], "boxes": [...]}, ...]
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 계산 효율을 위해 배치 데이터를 펼침 (Flatten)
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # 2. 모든 타겟을 하나로 합침
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 3. Cost Matrix 계산
        # Classification cost (Softmax 결과 사용)
        cost_class = -out_prob[:, tgt_ids]

        # L1 cost (Box coordinate distance)
        cost_bbox = torch.cdist(out_prob, tgt_bbox, p=1)

        # GIoU cost (Box overlap distance - 여기서는 단순화 위해 생략하거나 별도 함수 호출)
        # cost_giou = ... 

        # 최종 Cost Matrix (B*Q, Total_GT)
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        # 4. Scipy의 linear_sum_assignment를 이용한 최적 매칭 (Hungarian Algorithm)
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]