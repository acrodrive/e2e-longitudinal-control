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
    device = Config.device
    is_cuda = device.type == 'cuda'

    batch_size = Config.batch_size
    epochs = Config.epochs
    fpn_out_channels = Config.fpn_out_channels
    num_classes = Config.num_classes
    weight_decay = Config.weight_decay

    lr = Config.lr
    backbone_lr = lr * 0.1
    head_lr = lr
    
    TRAIN_JSON_PATH = Config.TRAIN_JSON_PATH
    TRAIN_IMG_DIR = Config.TRAIN_IMG_DIR
    VAL_JSON_PATH = Config.VAL_JSON_PATH
    VAL_IMG_DIR = Config.VAL_IMG_DIR
    CHECKPOINT_PATH = Config.CHECKPOINT_PATH

    print(f"Device: {device}, Batch size: {batch_size}, Epochs: {epochs}")
    print(f"Learning rates of the backbone and head: {backbone_lr}, {head_lr}")
    print(f"Json path of train data: {TRAIN_JSON_PATH}")
    print(f"Json path of val data: {VAL_JSON_PATH}")
    print(f"Image directory of train data: {TRAIN_IMG_DIR}")
    print(f"Image directory of val data: {VAL_IMG_DIR}")
    print(f"Checkpoint file: {CHECKPOINT_PATH}")
    
    wandb.init(
        project="BDD-Detection-Project",
        config={
            "learning_rate": Config.lr,
            "epochs": Config.epochs,
            "batch_size": Config.batch_size,
            "backbone": "ResNet50-FPN",
            "optimizer": "AdamW"
        }
    )

    # MODEL
    backbone = ResNetFPN(out_channels=fpn_out_channels).to(device)
    head = DetectionHead(num_classes=num_classes).to(device)
    
    # COMPILE TO ACCELERATE
    if is_cuda:
        try:
            backbone = torch.compile(backbone)
            head = torch.compile(head)
            print("=> Model compilation enabled.")
        except Exception as e:
            print(f"=> Compilation failed, proceeding without it: {e}")

    # LOSS
    criterion = MultiLevelDetectionLoss().to(device)

    # OPTIMIZER
    optimizer = torch.optim.AdamW([
        {'params': backbone.parameters(), 'lr': backbone_lr},
        {'params': head.parameters(), 'lr': head_lr}
    ], weight_decay=weight_decay)
    
    # SCHEDULER
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # SCALER
    scaler = torch.cuda.amp.GradScaler() if is_cuda else None
    
    # METRICS
    metrics = MetricsCalculator()
    metrics_val = MAPCalculator(device=device)
    
    # LOAD WEIGHTS
    start_epoch = load_model_weights(backbone, head, optimizer, scaler, CHECKPOINT_PATH)

    # LOAD DATASET
    train_transform = get_train_transforms(bbox_format=Config.bbox_format)
    train_dataset = BDDDataset(json_path=TRAIN_JSON_PATH, img_dir=TRAIN_IMG_DIR, transform=train_transform, num_classes=num_classes, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=is_cuda, collate_fn=collate_fn, persistent_workers=False)
    
    val_dataset = BDDDataset(json_path=VAL_JSON_PATH, img_dir=VAL_IMG_DIR, transform=None, num_classes=num_classes, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=is_cuda, collate_fn=collate_fn_for_validation, persistent_workers=False)

    print(f"Starting training on {device} (FP16: {is_cuda})...")

    for epoch in range(start_epoch, start_epoch + epochs):
        avg_loss = train_one_epoch(backbone, head, train_loader, criterion, optimizer, scheduler, scaler, device, metrics, epoch, start_epoch + Config.epochs)
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