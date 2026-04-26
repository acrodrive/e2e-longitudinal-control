import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import os
import math

class BDDDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None, num_classes=10, max_retries=20):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes
        self.max_retries = int(max_retries)
        self.strides = [8, 16, 32]  # P3, P4, P5 stride
        self.num_images = 0
        self.num_dropped_images = 0
        
        # BDD100K 클래스 매핑 (아래는 임의의 순서로 추후 변경이 필요한 경우 변경할 것)
        self.cat_to_id = {
            'pedestrian': 0, 'rider': 1, 'bike': 2, 'motor': 3,
            'car': 4, 'bus': 5, 'truck': 6, 
            'traffic light': 7, 'traffic sign': 8,
            'train': 9
        }

    def __len__(self):
        return len(self.data)

    def _gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(max(0, b1 ** 2 - 4 * a1 * c1))
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(max(0, b2 ** 2 - 4 * a2 * c2))
        r2  = (b2 - sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(max(0, b3 ** 2 - 4 * a3 * c3))
        r3  = (-b3 + sq3) / (2 * a3)
        
        return max(0, int(min(r1, r2, r3)))

    def _draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self._gaussian_kernel(radius, sigma=diameter / 6)

        gaussian[radius, radius] = 1.0

        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def _gaussian_kernel(self, radius, sigma):
        size = 2 * radius + 1
        x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g

    def _read_image_rgb(self, img_path: str):
        if not img_path:
            raise ValueError("Empty image path.")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise OSError(f"Failed to decode image (cv2.imread returned None): {img_path}")

        try:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            raise OSError(f"Failed to convert image to RGB: {img_path}") from e

    def __getitem__(self, idx): #idx는 데이터셋 전체 중에서 n번째 사진
        last_err = None
        
        try:
            item = self.data[idx]

            #영상의 파일명을 가져옴
            name = item.get('name')
            if not name:
                raise KeyError(f"Missing 'name' field in annotation at index {idx}.")

            img_path = None
            img_paths = []

            img_paths.append(os.path.join(self.img_dir, name))
            img_paths.append(os.path.join(self.img_dir, "trainA", name))
            img_paths.append(os.path.join(self.img_dir, "trainB", name))

            for candidate in img_paths:
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if not img_path:
                raise FileNotFoundError(f"Image not found for any candidate paths: {img_paths}")

            image = self._read_image_rgb(img_path)

            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Unexpected image shape {getattr(image, 'shape', None)} for {img_path}")

            H, W, _ = image.shape
            
            if H <= 0 or W <= 0:
                raise ValueError(f"Invalid image size (H={H}, W={W}) for {img_path}")

            self.num_images += 1

            bboxes = []
            class_labels = []

            for obj in item.get('labels', []):
                if 'box2d' in obj and obj['category'] in self.cat_to_id:
                    b = obj['box2d']

                    bbox_ctx = (b['x1'] + b['x2']) / 2
                    bbox_cty = (b['y1'] + b['y2']) / 2
                    bbox_w = abs(b['x1'] - b['x2'])
                    bbox_h = abs(b['y1'] - b['y2'])

                    bbox_ctx_norm = bbox_ctx / W
                    bbox_cty_norm = bbox_cty / H
                    bbox_w_norm = bbox_w / W
                    bbox_h_norm = bbox_h / H

                    bboxes.append([bbox_ctx_norm, bbox_cty_norm, bbox_w_norm, bbox_h_norm])
                    class_labels.append(self.cat_to_id[obj['category']])

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            self.num_dropped_images += 1
            return None
        
        # Extraction of bbox and class in the json file.

        # Augmentation
        if self.transform: # The bbox type of the transform is yolo in this project.
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            # just tensorize with transform to be None
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 의도적으로 픽셀이 겹치도록 설정
        scale_ratios = {
            8:  [0, 10],    # stride 8의 0~8배 크기 (0~64px @1x)
            16: [4, 18],   # stride 16의 4~12배 크기 (64~192px @1x)
            32: [8, 1000]  # stride 32의 8배 이상 (256px+ @1x)
        }
        results = {}
        # Generate Gaussian Distribution of Ground Truth on the feature maps.
        for s in self.strides:
            _, H, W = image.shape
            h_f, w_f = math.ceil(H / s), math.ceil(W / s)

            hm = np.zeros((self.num_classes, h_f, w_f), dtype=np.float32)
            regs = np.zeros((4, h_f, w_f), dtype=np.float32)
            mask = np.zeros((1, h_f, w_f), dtype=np.uint8)

            current_ratio_range = scale_ratios[s]
            
            for i, box in enumerate(bboxes):
                aug_ctx_norm, aug_cty_norm, aug_w_norm, aug_h_norm = box

                aug_ctx = aug_ctx_norm * w_f
                aug_cty = aug_cty_norm * h_f
                aug_w = aug_w_norm * w_f
                aug_h = aug_h_norm * h_f

                scale_in_feature_map = np.sqrt(aug_w * aug_h)
                if not (current_ratio_range[0] <= scale_in_feature_map < current_ratio_range[1]):
                    continue

                cls_id = int(class_labels[i]) # The class_labels[i] possibly appear the type of like numpy.int32
                
                radius = self._gaussian_radius((aug_h, aug_w)) # 원본 픽셀 기준
                radius = max(1, int(radius)) # P5의 경우 반지름을 1 이상으로 설정해야 할 수도 있음

                ix = np.clip(int(aug_ctx), 0, w_f - 1)
                iy = np.clip(int(aug_cty), 0, h_f - 1)
                
                if 0 <= ix < w_f and 0 <= iy < h_f:
                    self._draw_gaussian(hm[cls_id], (ix, iy), radius) # 정수 좌표 전달
                    regs[0, iy, ix] = aug_w
                    regs[1, iy, ix] = aug_h
                    regs[2, iy, ix] = aug_ctx - ix # offset_x
                    regs[3, iy, ix] = aug_cty - iy # offset_y
                    mask[0, iy, ix] = 1

            results[f'hm_s{s}'] = torch.from_numpy(hm)
            results[f'reg_s{s}'] = torch.from_numpy(regs)
            results[f'mask_s{s}'] = torch.from_numpy(mask)

        return image, results