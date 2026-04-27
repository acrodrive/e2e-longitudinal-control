import os
import torch
import cv2
import numpy as np
from PIL import Image

from projects.CNN.config import Config
from lib.models.resnet_fpn import ResNetFPN
from lib.models.head import DetectionHead
from lib.data.augmentation import get_inference_transforms # 혹은 별도의 val_transforms

def load_inference_model(checkpoint_path, device):
    # 1. 모델 구조 정의
    backbone = ResNetFPN(out_channels=Config.fpn_out_channels).to(device)
    head = DetectionHead(num_classes=Config.num_classes).to(device)
    
    # 2. 가중치 로드
    print(f"=> Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # train.py 저장 방식에 맞춰 key 로드
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    
    backbone.eval()
    head.eval()
    return backbone, head

@torch.no_grad()
def run_inference(image_path, backbone, head, transform, device):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    # transform은 [C, H, W] 형태의 Tensor를 반환해야 함
    input_tensor = transform(image).unsqueeze(0).to(device) 
    
    # 추론
    p3, p4, p5 = backbone(input_tensor)
    predictions = head(p3, p4, p5)
    
    return predictions

def main():
    device = Config.device
    checkpoint_path = os.path.join("checkpoints", "last_model.pth.tar")
    image_path = "test_image.jpg" # 추론할 이미지 경로
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found!")
        return

    # 1. 모델 준비
    backbone, head = load_inference_model(checkpoint_path, device)
    
    # 2. 전처리 (학습 때와 동일한 정규화 적용 권장)
    transform = get_inference_transforms() 

    # 3. 추론 실행
    preds = run_inference(image_path, backbone, head, transform, device)
    
    # 4. 결과 해석 (Post-processing)
    # DetectionHead의 출력 구조에 따라 NMS(Non-Maximum Suppression) 등이 필요할 수 있습니다.
    print("Inference complete.")
    print(f"Output shape example: {preds[0].shape if isinstance(preds, list) else preds.shape}")
    
    # 그림 그리고 이미지 저장까지 하자

if __name__ == "__main__":
    main()