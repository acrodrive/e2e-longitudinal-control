import os
import torch
from torch.utils.data import DataLoader

from projects.detector.models.obejct_detector import ObjectDetector
from projects.detector.models.loss import DetectionLoss
from projects.detector.models.matcher import HungarianMatcher
from projects.detector.training.optimizer import get_optimizer
from projects.detector.utils.logger import Logger
from projects.detector.data.bdd_loader import BDD100KDataset, collate_fn
from projects.detector.data.augmentations import get_train_transforms, get_val_transforms
from projects.detector.utils.metrics import BDD100KEvaluator

from config import Config

def train_one_epoch(model, loader, criterion, optimizer, device, logger, epoch):
    model.train()
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        # targets 전처리 로직 필요...
        
        outputs = model(torch.stack(images))
        loss_dict = criterion(outputs, targets)
        
        total_loss = sum(loss_dict.values())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            logger.log_metrics(loss_dict, epoch * len(loader) + i)

    

@torch.no_grad()
def validate(model, val_loader, device, config):
    model.eval()
    # 검증셋용 Evaluator 생성
    evaluator = BDD100KEvaluator(config.val_ann)

    for images, targets in val_loader:
        # images: list of tensors, targets: list of dicts
        images = torch.stack(images).to(device)
        
        outputs = model(images)
        
        # 결과 누적
        evaluator.update(outputs, targets)

    # 최종 mAP 계산
    stats = evaluator.summarize()
    return stats

def main():
    
    config = Config()
    device = torch.device(config.device)

    # 1. 모델 아키텍처 불러오기
    model = ObjectDetector(config).to(device)

    # 2. 학습 데이터셋 기본 설정
    train_dataset = BDD100KDataset(
        root=config.data_root,
        ann_file=config.train_ann,
        transform=get_train_transforms(config.img_size)
        )

    # 3. 실제 학습 데이터셋 (input_image, 정답 레이블)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        collate_fn=collate_fn # BDD100K처럼 타겟 크기가 다를 때 필수
        )

    val_dataset = BDD100KDataset(
        root=config.data_root, 
        ann_file=config.val_ann, 
        transform=get_val_transforms(config.img_size) # 검증용 증강(Resize만 수행)
        )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, # 평가 시에는 순서대로 읽어도 됨
        num_workers=config.num_workers,
        collate_fn=collate_fn
        )

    matcher = HungarianMatcher(cost_class=config.cost_class, cost_bbox=config.cost_bbox) ########################## 이 인자가 뭐지?
    criterion = DetectionLoss(num_classes=config.num_classes, matcher=matcher).to(device)

    optimizer, scheduler = get_optimizer()
    logger = Logger(log_dir=f"./logs/{config.exp_name}")
    

    """for epoch in range(config.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, logger, epoch)
        scheduler.step() # 에폭 끝날 때마다 스케줄러 업데이트

        # 여기에 검증(Validation) 및 체크포인트 저장 로직을 추가해야 함"""
    
    best_map = 0.0  # 최고 성적 기록용
    save_dir = f"./checkpoints/{config.exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config.epochs):
        # 1. 학습
        train_one_epoch(model, train_loader, criterion, optimizer, device, logger, epoch)
        
        # 2. 검증 (Validation)
        print(f"--- Epoch {epoch} Validation ---")
        val_metrics = validate(model, val_loader, device) # 별도 구현 필요
        current_map = val_metrics['mAP']
        
        # 3. 로그 기록
        logger.log_metrics(val_metrics, epoch)
        
        # 4. 스케줄러 업데이트
        scheduler.step()

        # 5. 체크포인트 저장 (Checkpoint Saving)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_map': best_map,
            'config': config
        }
        
        # 매 에폭마다 마지막 상태 저장 (Last Checkpoint)
        torch.save(checkpoint, os.path.join(save_dir, "last_model.pth"))
        
        # 최고 성능일 때만 저장 (Best Checkpoint)
        if current_map > best_map:
            best_map = current_map
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            print(f"*** Best Model Saved! mAP: {best_map:.4f} ***")
