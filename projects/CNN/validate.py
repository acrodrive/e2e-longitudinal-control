import torch
from torch.utils.data import DataLoader
from projects.CNN.config import Config
from lib.models.resnet_fpn import ResNetFPN
from lib.models.head import DetectionHead
from lib.models.loss import MultiLevelDetectionLoss
from lib.data.bdd_loader import BDDDataset
from lib.data.augmentation import get_val_transforms
from lib.engine.val_loop import validate_one_epoch
from lib.utils.metrics import MetricsCalculator
from lib.utils.utils import collate_fn

def evaluate():
    device = Config.device
    
    # 1. 모델 로드
    backbone = ResNetFPN(out_channels=Config.fpn_out_channels).to(device)
    head = DetectionHead(num_classes=Config.num_classes).to(device)
    
    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    
    # 2. 데이터셋 (Validation용)
    val_transform = get_val_transforms(bbox_format=Config.bbox_format)
    # JSON_PATH와 IMG_DIR를 Config에 val 전용으로 추가하거나, 아래처럼 직접 입력 가능합니다.
    val_dataset = BDDDataset(
        json_path="datasets/archive/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json", 
        img_dir="datasets/archive/bdd100k/bdd100k/images/100k/val", 
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    criterion = MultiLevelDetectionLoss().to(device)
    metrics = MetricsCalculator()

    print("Starting Evaluation...")
    avg_loss, stats = validate_one_epoch(backbone, head, val_loader, criterion, device, metrics, 0)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Avg Confidence: {stats['avg_pos_conf']:.4f}")
    print(f"Avg Pixel MAE: {stats['avg_pixel_mae']:.4f}")

if __name__ == "__main__":
    evaluate()