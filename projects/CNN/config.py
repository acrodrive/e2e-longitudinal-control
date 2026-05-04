import torch

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    batch_size = 16
    lr = 5e-4
    weight_decay = 1e-4 # 학습이 진행되고 나면 1e-3나 5e-4로 높이기
    epochs = 20
    mAP_threshold = 0.05

    fpn_out_channels = 256

    TRAIN_JSON_PATH = "datasets/archive/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    TRAIN_IMG_DIR = "datasets/archive/bdd100k/bdd100k/images/100k/train"    
    #VAL_JSON_PATH = "datasets/archive/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    VAL_JSON_PATH = "datasets/bdd100k/labels/bdd100k_labels_images_val.json"
    #VAL_IMG_DIR = "datasets/archive/bdd100k/bdd100k/images/100k/val"
    VAL_IMG_DIR = "datasets/bdd100k/images/100k/val"
    CHECKPOINT_PATH = None # "checkpoints/last_model.pth.tar"

    img_W = 1280
    img_H = 720

    # Mixed Precision Training (FP16) 사용 여부
    use_amp = True

    bbox_format = 'yolo' # other options: 'pascal_voc', 'albumentations', 'coco'