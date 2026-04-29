import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import os
import math

class BDDDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None, num_classes=10, max_retries=20, mode='train'):
        # 1. 원본 JSON 데이터 로드
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes
        self.max_retries = int(max_retries)
        self.strides = [8, 16, 32]
        self.num_images = 0
        self.num_dropped_images = 0
        self.mode = mode # 'train', 'val', 'test' 중 선택

        # BDD100K 클래스 매핑
        self.cat_to_id = {
            'pedestrian': 0, 'rider': 1, 'bike': 2, 'motor': 3,
            'car': 4, 'bus': 5, 'truck': 6, 
            'traffic light': 7, 'traffic sign': 8,
            'train': 9
        }

        self.file_to_path = {}
        # 2. 실제 폴더(trainA, trainB 등) 내의 파일 리스트 스캔
        # 사용자가 분류한 하위 폴더 리스트
        if self.mode == 'train':
            self.sub_dirs = ['trainA', 'trainB', 'testA', 'testB']

            
            for sub in self.sub_dirs:
                sub_path = os.path.join(self.img_dir, sub)
                if os.path.exists(sub_path):
                    for f_name in os.listdir(sub_path):
                        # 파일명: 실제 경로 매핑 저장
                        self.file_to_path[f_name] = os.path.join(sub_path, f_name)

            # 3. 폴더에 존재하는 파일들만 JSON 데이터에서 필터링
            # JSON의 'name'이 실제 폴더 안에 존재하는 경우만 self.data에 유지
            self.data = [item for item in full_data if item.get('name') in self.file_to_path]
            
            print(f"Total train images found in folders: {len(self.file_to_path)}")
            print(f"Total valid annotations matched: {len(self.data)}")
        elif self.mode == 'val':
            if os.path.exists(self.img_dir):
                for f_name in os.listdir(self.img_dir):
                    self.file_to_path[f_name] = os.path.join(self.img_dir, f_name)

            self.data = [item for item in full_data if item.get('name') in self.img_dir]

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
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise OSError(f"Failed to decode image: {img_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            name = item.get('name')
            
            # 이미 __init__에서 검증된 경로를 즉시 가져옴
            img_path = self.file_to_path.get(name)
            if not img_path:
                raise FileNotFoundError(f"Image {name} not found in pre-scanned paths.")

            image = self._read_image_rgb(img_path)
            H, W, _ = image.shape
            self.num_images += 1

            bboxes = []
            class_labels = []

            for obj in item.get('labels', []):
                if 'box2d' in obj and obj['category'] in self.cat_to_id:
                    b = obj['box2d']
                    # 중심점 및 크기 정규화 계산
                    ctx = (b['x1'] + b['x2']) / 2 / W
                    cty = (b['y1'] + b['y2']) / 2 / H
                    bw = abs(b['x1'] - b['x2']) / W
                    bh = abs(b['y1'] - b['y2']) / H

                    bboxes.append([ctx, cty, bw, bh])
                    class_labels.append(self.cat_to_id[obj['category']])

        except Exception as e:
            # 에러 발생 시 None 반환 (DataLoader에서 collate_fn 처리가 필요할 수 있음)
            self.num_dropped_images += 1
            return None
        
        # Augmentation 및 Feature Map 생성 로직은 기존과 동일하게 유지
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        scale_ratios = {8: [0, 10], 16: [4, 18], 32: [8, 1000]} # 이거 일부러 이랬긴 했거든 근데 이게 이점이 있을까?
        results = {}


        if self.mode == 'train':
            for s in self.strides:
                _, H, W = image.shape
                h_f, w_f = math.ceil(H / s), math.ceil(W / s)
                hm = np.zeros((self.num_classes, h_f, w_f), dtype=np.float32)
                regs = np.zeros((4, h_f, w_f), dtype=np.float32)
                mask = np.zeros((1, h_f, w_f), dtype=np.uint8)
                
                current_range = scale_ratios[s]
                for i, box in enumerate(bboxes):
                    ctx_norm, cty_norm, w_norm, h_norm = box
                    # 피처맵 크기에 맞게 스케일 조정
                    f_ctx, f_cty = ctx_norm * w_f, cty_norm * h_f
                    f_w, f_h = w_norm * w_f, h_norm * h_f

                    if not (current_range[0] <= np.sqrt(f_w * f_h) < current_range[1]):
                        continue

                    radius = max(1, int(self._gaussian_radius((f_h, f_w))))
                    ix, iy = int(f_ctx), int(f_cty)
                    
                    if 0 <= ix < w_f and 0 <= iy < h_f:
                        self._draw_gaussian(hm[int(class_labels[i])], (ix, iy), radius)
                        regs[0, iy, ix] = f_w
                        regs[1, iy, ix] = f_h
                        regs[2, iy, ix] = f_ctx - ix
                        regs[3, iy, ix] = f_cty - iy
                        mask[0, iy, ix] = 1

                results[f'hm_s{s}'] = torch.from_numpy(hm)
                results[f'reg_s{s}'] = torch.from_numpy(regs)
                results[f'mask_s{s}'] = torch.from_numpy(mask)

            return image, results
        else:
            target = {
                "boxes": torch.as_tensor(bboxes, dtype=torch.float32),
                "labels": torch.as_tensor(class_labels, dtype=torch.int64),
                "image_name": self.data[idx].get('name')
            }
            return image, target