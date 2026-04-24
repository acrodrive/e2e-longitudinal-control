import torch

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    batch_size = 16
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 100

    fpn_out_channels = 256

    JSON_PATH = "data/bdd100k/labels/bdd100k_labels_images_train.json"
    IMG_DIR = "data/bdd100k/images/100k/train"
    CHECKPOINT_PATH = None # "checkpoint/"

    img_W = 1280
    img_H = 720

    # Mixed Precision Training (FP16) 사용 여부
    use_amp = True

    bbox_format = 'yolo' # other options: 'pascal_voc', 'albumentations', 'coco'