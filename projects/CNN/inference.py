import torch
import cv2
import numpy as np
from projects.CNN.config import Config
from lib.models.resnet_fpn import ResNetFPN
from lib.models.head import DetectionHead
from lib.data.augmentation import get_inference_transforms
from lib.utils.utils import post_process
from visualize import draw_detections

def run_inference(image_path, model_path):
    device = Config.device
    class_names = [
        'pedestrian', 'rider', 'bike', 'motor', 'car', 
        'bus', 'truck', 'traffic light', 'traffic sign', 'train'
    ]
    strides = [8, 16, 32]

    # 1. 모델 로드
    backbone = ResNetFPN(out_channels=Config.fpn_out_channels).to(device)
    head = DetectionHead(num_classes=Config.num_classes).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    # torch.compile을 사용했다면 키 이름에 '_orig_mod.'가 붙을 수 있으므로 처리
    backbone_state = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['backbone_state_dict'].items()}
    head_state = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['head_state_dict'].items()}
    
    backbone.load_state_dict(backbone_state)
    head.load_state_dict(head_state)
    backbone.eval()
    head.eval()

    # 2. 이미지 준비
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    transform = get_inference_transforms()
    # transform은 이미지 크기 조정을 포함하지 않으므로 필요시 Resize 추가 가능
    input_tensor = transform(image=orig_img)['image'].unsqueeze(0).to(device)

    # 3. 추론
    with torch.no_grad():
        p3, p4, p5 = backbone(input_tensor)
        preds = head(p3, p4, p5)
        
        pred_hms = [p[0] for p in preds]
        pred_regs = [p[1] for p in preds]

    # 4. 후처리 (Decoding)
    detections = post_process(pred_hms, pred_regs, strides, threshold=0.3)

    # 5. 시각화
    vis_result = draw_detections(orig_img, detections, class_names)
    
    # 결과 출력 및 저장
    cv2.imshow("Inference Result", vis_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("result.jpg", vis_result)

if __name__ == "__main__":
    # 실행 전 체크포인트 경로와 이미지 경로 수정 필요
    MODEL_PATH = "checkpoints/last_model.pth.tar" 
    TEST_IMAGE = "data/archive/bdd100k/bdd100k/images/100k/train/0000f77c-62c2a288.jpg"
    
    run_inference(TEST_IMAGE, MODEL_PATH)