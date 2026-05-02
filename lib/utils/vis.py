import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# BDD100K 클래스 ID -> 이름 매핑
ID_TO_CAT = {
    0: 'pedestrian', 1: 'rider', 2: 'bike', 3: 'motor',
    4: 'car', 5: 'bus', 6: 'truck', 
    7: 'traffic light', 8: 'traffic sign', 9: 'train'
}

def visualize_predictions(image_rgb, detections, threshold=0.3):
    """
    원본 이미지 위에 예측된 바운딩 박스와 클래스 이름, Confidence Score를 시각화합니다.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    for det in detections:
        score = det['score'].item()
        if score < threshold:
            continue
            
        box = det['box'].cpu().numpy()
        cls_id = det['class_id'].item()
        cls_name = ID_TO_CAT.get(cls_id, str(cls_id))
        
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        
        # 바운딩 박스 그리기
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2, edgecolor='red', fill=False)
        ax.add_patch(rect)
        
        # 라벨 텍스트 추가
        ax.text(x1, y1 - 5, f'{cls_name} {score:.2f}', color='red', fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        
    plt.axis('off')
    plt.tight_layout()
    return fig

def visualize_predicted_heatmaps(image_rgb, pred_hms, strides):
    """
    모델이 예측한 히트맵(pred_hms)을 스트라이드별로 시각화합니다.
    debugging.py의 visualize_targets 구조를 참고했습니다.
    """
    fig, axes = plt.subplots(len(strides), 2, figsize=(12, 5 * len(strides)))
    
    for idx, s in enumerate(strides):
        # pred_hms[idx] shape: [B, C, H, W] -> 배치 1이므로 [C, H, W] 가져오기
        hm = pred_hms[idx][0].detach().cpu().numpy()
        
        # 모든 클래스 중 가장 높은 활성화 값을 가진 채널로 압축 [H, W]
        hm_max = np.max(hm, axis=0) 
        
        # 원본 이미지 크기로 Resize
        hm_resized = cv2.resize(hm_max, (image_rgb.shape[1], image_rgb.shape[0]))
        
        # (1) 원본 이미지 + 히트맵 오버레이
        axes[idx, 0].imshow(image_rgb)
        axes[idx, 0].imshow(hm_resized, alpha=0.5, cmap='jet')
        axes[idx, 0].set_title(f'Predicted Heatmap Overlay (Stride {s})', fontsize=15)
        axes[idx, 0].axis('off')

        # (2) 순수 예측 히트맵
        axes[idx, 1].imshow(hm_max, cmap='viridis')
        axes[idx, 1].set_title(f'Predicted Raw Heatmap (Stride {s})', fontsize=15)
        axes[idx, 1].axis('off')

    plt.tight_layout()
    return fig