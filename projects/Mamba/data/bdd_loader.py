import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import os

class BDD100KDataset(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        # BDD100K 클래스 매핑 (예시)
        self.label_map = {'pedestrian': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7, 'traffic light': 8, 'traffic sign': 9}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.root, item['name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        labels = []
        for obj in item.get('labels', []):
            if 'box2d' in obj and obj['category'] in self.label_map:
                box = obj['box2d']
                # COCO format: [x_min, y_min, width, height]
                bboxes.append([box['x1'], box['y1'], box['x2']-box['x1'], box['y2']-box['y1']])
                labels.append(self.label_map[obj['category']])

        if self.transform:
            augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = augmented['image']
            # Bbox를 [0, 1]로 정규화하는 로직 필요 (DETR 기준)
            target = {
                'boxes': torch.as_tensor(augmented['bboxes'], dtype=torch.float32),
                'labels': torch.as_tensor(augmented['labels'], dtype=torch.long)
            }
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))