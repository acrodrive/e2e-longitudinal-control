from projects.CNN.config import Config
from lib.data.bdd_loader import BDDDataset
from lib.data.augmentation import get_train_transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import matplotlib.patches as patches

def visualize_targets(image_show, target, strides):
    fig, axes = plt.subplots(len(strides), 4, figsize=(24, 5 * len(strides)))
    
    for idx, s in enumerate(strides):
        # 데이터 가져오기
        hm = target[f'hm_s{s}'].detach().cpu().numpy()
        reg = target[f'reg_s{s}'].detach().cpu().numpy()
        mask = target[f'mask_s{s}'].detach().cpu().numpy()
        
        # [C, Hf, Wf] -> [Hf, Wf] (전체 클래스 응답 통합)
        hm_max = np.max(hm, axis=0)
        
        # --- (1) Original + Heatmap Overlay ---
        # 히트맵을 원본 1280x720 크기로 확대
        hm_resized = cv2.resize(hm_max, (image_show.shape[1], image_show.shape[0]))
        axes[idx, 0].imshow(image_show)
        # 높은 응답 구역을 빨간색으로 표시 (alpha로 투명도 조절)
        axes[idx, 0].imshow(hm_resized, alpha=0.4, cmap='jet')
        axes[idx, 0].set_title(f'Stride {s}: Image Overlay', fontsize=15)
        axes[idx, 0].axis('off')

        # --- (2) Pure Heatmap ---
        axes[idx, 1].imshow(hm_max, cmap='viridis')
        axes[idx, 1].set_title(f'Stride {s}: Raw Heatmap', fontsize=15)
        axes[idx, 1].axis('off')

        # --- (3) Mask (Object Centers) ---
        # 점이 너무 작아서 잘 안 보일 수 있으므로, mask 위치만 흰색으로 강조
        axes[idx, 2].imshow(mask[0], cmap='gray')
        axes[idx, 2].set_title(f'Stride {s}: Center Points', fontsize=15)
        axes[idx, 2].axis('off')

        # --- (4) Regression Value (Width) ---
        # 마스크가 있는 위치의 '폭(width)' 값만 시각화
        reg_w = reg[0] * mask[0]
        im_reg = axes[idx, 3].imshow(reg_w, cmap='magma')
        axes[idx, 3].set_title(f'Stride {s}: Reg Width Value', fontsize=15)
        axes[idx, 3].axis('off')
        fig.colorbar(im_reg, ax=axes[idx, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def visualize_bbox_from_target(image_tensor, target, strides):
    # 1. 이미지 변환 [C, H, W] -> [H, W, C] (0~255 uint8)
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    H, W, _ = img.shape

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)

    # 스트라이드마다 색상을 다르게 설정 (구분용)
    colors = {8: 'r', 16: 'g', 32: 'b'}

    for s in strides:
        mask = target[f'mask_s{s}'].numpy()[0]  # [Hf, Wf]
        reg = target[f'reg_s{s}'].numpy()       # [4, Hf, Wf] (w, h, offset_x, offset_y)
        
        # mask가 1인 곳(객체 중심점)의 인덱스를 찾음
        ys, xs = np.where(mask == 1)

        for y, x in zip(ys, xs):
            # 1. reg에서 값 추출
            f_w = reg[0, y, x]
            f_h = reg[1, y, x]
            offset_x = reg[2, y, x]
            offset_y = reg[3, y, x]

            # 2. 피처맵 좌표 -> 원본 이미지 좌표로 복원
            # 중심점 복원: (정수 좌표 + 오프셋) * 스트라이드
            ctx = (x + offset_x) * s
            cty = (y + offset_y) * s
            
            # 크기 복원: 피처맵 크기 * 스트라이드
            bw = f_w * s
            bh = f_h * s

            # 3. 박스 좌상단(x1, y1) 계산
            x1 = ctx - bw / 2
            y1 = cty - bh / 2

            # 4. 그리기
            rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2, 
                                     edgecolor=colors.get(s, 'yellow'), fill=False)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'S{s}', color=colors.get(s, 'yellow'), fontweight='bold')

    plt.title("Restored Bounding Boxes from Targets")
    plt.axis('off')
    plt.show()
    
def main():
    print("Loading Dataset ...")
    TRAIN_JSON_PATH = "datasets/bdd100k/labels/bdd100k_labels_images_train.json"
    TRAIN_IMG_DIR = "datasets/bdd100k/images/100k/train"
    # train_transform = get_train_transforms(bbox_format=Config.bbox_format)
    train_dataset = BDDDataset(json_path=TRAIN_JSON_PATH, img_dir=TRAIN_IMG_DIR, transform=None, num_classes=Config.num_classes, mode='train')
    print("Dataset loaded")
    
    strides = [8, 16, 32]

    for i in range(5, 11):
        image_tensor, target = train_dataset[i]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_show = (image_np * 255).astype(np.uint8)
        
        visualize_targets(image_show, target, strides)
        
        # visualize_bbox_from_target(image_tensor, target, strides)
    
    
if __name__ == '__main__':
    main()