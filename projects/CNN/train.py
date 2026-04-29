import wandb
import torch
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader
from projects.CNN.config import Config
from lib.models.resnet_fpn import ResNetFPN
from lib.models.head import DetectionHead
from lib.models.loss import MultiLevelDetectionLoss
from lib.data.bdd_loader import BDDDataset
from lib.data.augmentation import get_train_transforms
from lib.engine.train_loop import train_one_epoch
from lib.engine.val_loop import validate_with_map
from lib.utils.metrics import MetricsCalculator, MAPCalculator
from lib.utils.utils import save_checkpoint, load_model_weights, collate_fn, collate_fn_for_validation

def main():
    #region 1. 기본 설정
    device = Config.device
    is_cuda = device.type == 'cuda'

    batch_size = Config.batch_size
    lr = Config.lr
    weight_decay = Config.weight_decay
    epochs = Config.epochs
    fpn_out_channels = Config.fpn_out_channels
    num_classes = Config.num_classes
    
    TRAIN_JSON_PATH = Config.TRAIN_JSON_PATH
    TRAIN_IMG_DIR = Config.TRAIN_IMG_DIR
    VAL_JSON_PATH = Config.VAL_JSON_PATH
    VAL_IMG_DIR = Config.VAL_IMG_DIR
    
    CHECKPOINT_PATH = Config.CHECKPOINT_PATH
    
    wandb.init(
        project="BDD-Detection-Project", # 프로젝트 이름
        config={
            "learning_rate": Config.lr,
            "epochs": Config.epochs,
            "batch_size": Config.batch_size,
            "backbone": "ResNet18-FPN",
            "optimizer": "AdamW"
        }
    )
    #endregion

    #region 2. 모델 구성
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
    #endregion
    
    #region 3. optim, FP16 scaler, matrics
    # old
    # all_params = list(backbone.parameters()) + list(head.parameters())
    # optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    # new
    backbone_lr = lr * 0.1
    head_lr = lr

    optimizer = torch.optim.AdamW([
        {'params': backbone.parameters(), 'lr': backbone_lr},
        {'params': head.parameters(), 'lr': head_lr}
    ], weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # CUDA일 때만 GradScaler 활성화
    scaler = torch.cuda.amp.GradScaler() if is_cuda else None
    
    metrics = MetricsCalculator()
    metrics_val = MAPCalculator(device=device)
    
    #endregion
    
    #region 4. 가중치 로드
    start_epoch = load_model_weights(backbone, head, optimizer, scaler, CHECKPOINT_PATH)
    #endregion

    #region 5. 데이터 준비
    train_transform = get_train_transforms(bbox_format=Config.bbox_format)
    train_dataset = BDDDataset(json_path=TRAIN_JSON_PATH, img_dir=TRAIN_IMG_DIR, transform=train_transform, num_classes=num_classes, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=is_cuda, collate_fn=collate_fn, persistent_workers=True)
    
    val_dataset = BDDDataset(json_path=VAL_JSON_PATH, img_dir=VAL_IMG_DIR, transform=None, num_classes=num_classes, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=is_cuda, collate_fn=collate_fn_for_validation, persistent_workers=True)

    print(f"Starting training on {device} (FP16: {is_cuda})...")

    # 학습 루프
    for epoch in range(start_epoch, epochs):
        avg_loss, stats = train_one_epoch(backbone, head, train_loader, criterion, optimizer, scheduler, scaler, device, metrics, epoch, start_epoch + Config.epochs)
        # 해결해야 할 문제: loss랑 metrics랑 지금 섞여있음. train.py, train_loop.py, loss.py에
        checkpoint_state = {
            'epoch': epoch + 1,
            'backbone_state_dict': backbone.state_dict(),
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        if is_cuda:
            checkpoint_state['scaler_state_dict'] = scaler.state_dict()
        
        validate_with_map(backbone, head, val_loader, device, metrics_val, epoch)
        save_checkpoint(checkpoint_state, epoch + 1)

if __name__ == "__main__":
    main()