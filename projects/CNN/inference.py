import torch
import cv2
import os
from projects.CNN.config import Config
from lib.models.resnet_fpn import ResNetFPN
from lib.models.head import DetectionHead
from lib.data.augmentation import get_inference_transforms
from lib.utils.utils import post_process
from lib.utils.vis import visualize_predictions, visualize_predicted_heatmaps, tensor_to_image_rgb
from lib.data.bdd_loader import BDDDataset

def main():
    device = Config.device
    
    # [사용자 설정 필요] 경로 지정
    checkpoint_path = "checkpoints/last_model.pth.tar" 
    image_path = "test_image.jpg"
    confidence_threshold = 0.3
    
    print(f"Loading model on {device}...")

    # inference.py의 main 함수 내 로드 부분 수정
    checkpoint = torch.load(checkpoint_path, map_location=device)

    backbone = ResNetFPN(out_channels=Config.fpn_out_channels).to(device)
    head = DetectionHead(num_classes=Config.num_classes).to(device)

    # --- Backbone 가중치 키 수정 ---
    backbone_state_dict = checkpoint['backbone_state_dict']
    new_backbone_state_dict = {}
    for k, v in backbone_state_dict.items():
        # '_orig_mod.' 접두어가 있다면 제거
        name = k.replace("_orig_mod.", "") 
        new_backbone_state_dict[name] = v
    backbone.load_state_dict(new_backbone_state_dict)

    # --- Head 가중치 키 수정 (Head도 컴파일했다면 필요) ---
    head_state_dict = checkpoint['head_state_dict']
    new_head_state_dict = {}
    for k, v in head_state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_head_state_dict[name] = v
    head.load_state_dict(new_head_state_dict)
    
    """# 1. 모델 초기화 및 가중치 로드
    backbone = ResNetFPN(out_channels=Config.fpn_out_channels).to(device)
    head = DetectionHead(num_classes=Config.num_classes).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])"""
    
    backbone.eval()
    head.eval()
    
    print("Loading Dataset ...")
    TRAIN_JSON_PATH = Config.TRAIN_JSON_PATH
    TRAIN_IMG_DIR = Config.TRAIN_IMG_DIR
    train_dataset = BDDDataset(json_path=TRAIN_JSON_PATH, img_dir=TRAIN_IMG_DIR, transform=None, num_classes=Config.num_classes, mode='train')
    print("Dataset loaded")

    img_tensor, target = train_dataset[0]
    image_rgb = tensor_to_image_rgb(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    #image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    #image_show = (image_np * 255).astype(np.uint8)
    
    #visualize_targets(image_show, target, strides)

    
    """# 2. 이미지 로드 및 전처리
    print(f"Processing image {image_path}...")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Inference용 Transform 적용 (Normalize 및 ToTensor)[cite: 1]
    transform = get_inference_transforms()
    transformed = transform(image=image_rgb)
    img_tensor = transformed['image'].unsqueeze(0).to(device) # [1, C, H, W] 배치 차원 추가"""

    gt_hms = []
    gt_regs = []
    for j in range(3):
        # target 값들은 [C, H, W] 이므로 unsqueeze(0)으로 [1, C, H, W]로 만듦
        gt_hm = target[f'hm_s{8*2**j}'].unsqueeze(0).to(device)
        gt_reg = target[f'reg_s{8*2**j}'].unsqueeze(0).to(device)
        
        gt_hms.append(gt_hm)
        gt_regs.append(gt_reg)
        
    # 3. 모델 추론
    print("Running inference...")
    with torch.no_grad():
        p3, p4, p5 = backbone(img_tensor)
        preds = head(p3, p4, p5) #[cite: 5]
        
        pred_hms = [p[0] for p in preds]
        pred_regs = [p[1] for p in preds]
        
        strides = [8, 16, 32]
        # post_process로 최종 디코딩된 바운딩 박스 추출
        detections = post_process(pred_hms, pred_regs, strides, threshold=confidence_threshold)[0]
 
        
    # 4. 결과 시각화 및 저장
    print("Visualizing and saving results...")
    
    # 바운딩 박스 결과
    fig_bbox = visualize_predictions(image_rgb, detections, threshold=confidence_threshold)
    fig_bbox.savefig("inference_bbox_result.png")
    
    # 히트맵 결과 (디버깅/분석용)
    fig_hm = visualize_predicted_heatmaps(image_rgb, pred_hms, strides)
    fig_hm.savefig("inference_heatmap_result.png")
    
    # GT 결과 시각화 (GT도 4D 텐서 리스트가 되었으므로 post_process 정상 작동)
    gt = post_process(gt_hms, gt_regs, strides, threshold=confidence_threshold)[0]
    gt_bbox = visualize_predictions(image_rgb, gt, threshold=confidence_threshold)
    gt_bbox.savefig("GT_bbox_result.png")
    
    gt_hm_fig = visualize_predicted_heatmaps(image_rgb, gt_hms, strides)
    gt_hm_fig.savefig("GT_heatmap_result.png")
    
    print("Done. Results saved as 'inference_bbox_result.png' and 'inference_heatmap_result.png'.")

if __name__ == '__main__':
    main()