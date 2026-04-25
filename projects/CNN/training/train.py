import os

import torch
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader

from projects.CNN.config import Config
from projects.CNN.models.resnet_fpn import ResNetFPN
from projects.CNN.models.head import DetectionHead
from projects.CNN.models.loss import MultiLevelDetectionLoss
from projects.CNN.data.bdd_loader import BDDDataset
from projects.CNN.data.augmentation import get_train_transforms

def save_checkpoint(state, epoch, folder="checkpoints"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # TODO: Include architecture name and timestamp in the filename.
    filename = os.path.join(folder, f"checkpoint_epoch_{epoch}.pth.tar")
    torch.save(state, filename)
    
    last_path = os.path.join(folder, "last_model.pth.tar")
    torch.save(state, last_path)
    print(f"--- Checkpoint saved: {filename} ---")

def load_model_weights(backbone, head, optimizer, scaler, checkpoint_path):
    start_epoch = 0
    if not checkpoint_path: 
        print("=> checkpoint : None")
        return start_epoch
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Scaler 상태도 복구하여 학습 연속성 유지
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"=> Resumed from epoch {start_epoch}")
    else:
        print("=> No checkpoint found. Starting from pre-trained backbone.")
    return start_epoch

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    if len(batch) == 0:
        return None
    
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    device = Config.device
    is_cuda = device.type == 'cuda'

    batch_size = Config.batch_size
    lr = Config.lr
    weight_decay = Config.weight_decay
    epochs = Config.epochs
    fpn_out_channels = Config.fpn_out_channels
    num_classes = Config.num_classes
    
    JSON_PATH = Config.JSON_PATH
    IMG_DIR = Config.IMG_DIR
    CHECKPOINT_PATH = Config.CHECKPOINT_PATH

    # 1. 모델 구성
    backbone = ResNetFPN(out_channels=fpn_out_channels).to(device)
    head = DetectionHead(num_classes=num_classes).to(device)
    
    # RTX 5090 성능 극대화를 위한 컴파일 (PyTorch 2.0+)
    if is_cuda:
        try:
            backbone = torch.compile(backbone)
            head = torch.compile(head)
            print("=> Model compilation enabled.")
        except Exception as e:
            print(f"=> Compilation failed, proceeding without it: {e}")

    criterion = MultiLevelDetectionLoss().to(device)
    
    # 2. 옵티마이저 및 FP16 스케일러
    all_params = list(backbone.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # CUDA일 때만 GradScaler 활성화
    scaler = torch.cuda.amp.GradScaler() if is_cuda else None
    # 3. 가중치 로드
    start_epoch = load_model_weights(backbone, head, optimizer, scaler, CHECKPOINT_PATH)

    # 4. 데이터 준비
    train_transform = get_train_transforms()
    train_dataset = BDDDataset(json_path=JSON_PATH, img_dir=IMG_DIR, transform=train_transform, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=is_cuda, collate_fn=collate_fn)

    print(f"Starting training on {device} (FP16: {is_cuda})...")

    for epoch in range(start_epoch, epochs):
        backbone.train()
        head.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(train_loader): # i = 0 to 17465, batch: image and GT
            if batch is None:
                continue
            imgs, targets_dict = batch
            imgs = imgs.to(device)
            targets_dict = {k: v.to(device) for k, v in targets_dict.items()}
            
            optimizer.zero_grad()
            
            if is_cuda:
                with amp.autocast(device_type=device.type):
                    p3, p4, p5 = backbone(imgs)
                    preds = head(p3, p4, p5)
                    loss = criterion(preds, targets_dict)
                
                # FP16 Backward & Step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU 환경 (Standard FP32)
                p3, p4, p5 = backbone(imgs)
                preds = head(p3, p4, p5)
                loss = criterion(preds, targets_dict)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
                print(f"Number of images in the dataset: {train_dataset.num_images}")
                print(f"Number of Dropped images in the dataset: {train_dataset.num_dropped_images}")

        
        #scheduler.step() 
        #current_lr = optimizer.param_groups[0]['lr']
        #print(f"Epoch {epoch} finished. Current LR: {current_lr:.6f}")

        # 6. 체크포인트 저장 (Scaler 상태 추가)
        checkpoint_state = {
            'epoch': epoch + 1,
            'backbone_state_dict': backbone.state_dict(),
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss / len(train_loader),
        }
        if is_cuda:
            checkpoint_state['scaler_state_dict'] = scaler.state_dict()
            
        save_checkpoint(checkpoint_state, epoch + 1)

if __name__ == "__main__":
    main()