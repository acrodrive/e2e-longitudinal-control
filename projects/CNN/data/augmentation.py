import albumentations as A
from albumentations.pytorch import ToTensorV2
from projects.CNN.config import Config

def get_train_transforms(img_size=(1280, 720)):
    return A.Compose([
        # 1. 기하학적 변환 (박스 좌표도 함께 변환됨)
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.PadIfNeeded(min_height=img_size[1], min_width=img_size[0], border_mode=0, p=1.0),
        A.RandomCrop(height=img_size[1], width=img_size[0], p=1.0),

        # 2. 색상 및 노이즈 변환
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # 3. 정규화 및 텐서 변환
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format=Config.bbox_format, label_fields=['class_labels'], clip=True, min_visibility=0.3))

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format=Config.bbox_format, label_fields=['class_labels'], clip=True, min_visibility=0.3))