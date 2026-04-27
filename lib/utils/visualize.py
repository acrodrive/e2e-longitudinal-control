import cv2
import numpy as np

def draw_detections(image, detections, class_names, threshold=0.3):
    """
    image: numpy array (H, W, 3) - RGB
    detections: list of dicts [{'box': [x1, y1, x2, y2], 'score': s, 'class_id': i}, ...]
    """
    vis_img = image.copy()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    for det in detections:
        score = det['score']
        if score < threshold:
            continue
            
        x1, y1, x2, y2 = map(int, det['box'])
        cls_id = det['class_id']
        label = f"{class_names[cls_id]} {score:.2f}"

        # 박스 그리기
        color = (0, 255, 0) # Green
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # 라벨 배경 및 텍스트
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis_img