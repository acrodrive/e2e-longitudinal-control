import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

class BDD100KEvaluator:
    def __init__(self, gt_json_path):
        """
        gt_json_path: BDD100K 검증셋 어노테이션 파일 경로
        """
        self.coco_gt = COCO(gt_json_path)
        self.results = []

    def update(self, outputs, targets):
        """
        한 배치의 결과를 누적합니다.
        outputs: 모델 출력 {"pred_logits": [B, Q, C], "pred_boxes": [B, Q, 4]}
        targets: 데이터 로더에서 제공한 원본 정보 (image_id 포함 필수)
        """
        # [batch_size, num_queries, num_classes]
        probs = outputs['pred_logits'].softmax(-1)
        scores, labels = probs[..., :-1].max(-1) # 배경 제외 최고 점수와 라벨
        boxes = outputs['pred_boxes'] # [B, Q, 4] (normalized cx,cy,w,h)

        for i, (s, l, b) in enumerate(zip(scores, labels, boxes)):
            img_id = targets[i]['image_id']
            img_info = self.coco_gt.loadImgs(img_id)[0]
            w, h = img_info['width'], img_info['height']

            # COCO 평가를 위해 normalized cxcywh -> absolute xywh 변환
            # b: [cx, cy, w, h] normalized
            b = b.cpu().numpy()
            abs_box = [
                (b[0] - 0.5 * b[2]) * w, # x_min
                (b[1] - 0.5 * b[3]) * h, # y_min
                b[2] * w,               # width
                b[3] * h                # height
            ]

            for score, label, box in zip(s.cpu().numpy(), l.cpu().numpy(), abs_box):
                if score < 0.05: # 아주 낮은 점수는 미리 필터링하여 계산 속도 향상
                    continue
                
                self.results.append({
                    "image_id": int(img_id),
                    "category_id": int(label), # BDD100K 실제 카테고리 ID 매핑 확인 필요
                    "bbox": [float(x) for x in box],
                    "score": float(score)
                })

    def summarize(self):
        """
        누적된 결과로 mAP를 계산합니다.
        """
        if not self.results:
            print("No predictions to evaluate.")
            return {"mAP": 0.0}

        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 주요 지표 추출 (mAP @ IoU=0.5:0.95)
        return {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2]
        }