import time
import torch
from torch.utils.data import DataLoader
from projects.CNN.config import Config
from lib.models.resnet_fpn import ResNetFPN
from lib.models.head import DetectionHead
from lib.data.bdd_loader import BDDDataset
from lib.utils.metrics import MAPCalculator
from lib.utils.utils import load_model_weights, collate_fn_for_validation, post_process, convert_to_metric_format

def main():
    # 1. 설정 및 장치 확인
    device = Config.device
    num_classes = Config.num_classes
    VAL_JSON_PATH = Config.VAL_JSON_PATH
    VAL_IMG_DIR = Config.VAL_IMG_DIR
    CHECKPOINT_PATH = Config.CHECKPOINT_PATH

    print(f"=== Evaluation Mode ===")
    print(f"Device: {device}")
    print(f"Loading weights from: {CHECKPOINT_PATH}")
    print(f"Evaluating on Validation Dataset: {VAL_IMG_DIR}")

    # 2. 모델 선언 및 평가 모드 설정
    backbone = ResNetFPN(out_channels=Config.fpn_out_channels).to(device)
    head = DetectionHead(num_classes=num_classes).to(device)
    
    # 가중치 로드 (평가 시에는 optimizer와 scaler가 필요 없으므로 None 전달)
    _ = load_model_weights(backbone, head, None, None, CHECKPOINT_PATH)
    
    backbone.eval()
    head.eval()

    # 3. 데이터셋 및 데이터로더 설정 (shuffle은 False로 변경)
    val_dataset = BDDDataset(
        json_path=VAL_JSON_PATH, 
        img_dir=VAL_IMG_DIR, 
        transform=None, 
        num_classes=num_classes, 
        mode='val'
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=device.type == 'cuda', 
        collate_fn=collate_fn_for_validation, 
        persistent_workers=False
    )

    # 4. 지표 계산기 초기화
    metrics_val = MAPCalculator(device=device)
    if hasattr(metrics_val, 'reset'):
        metrics_val.reset()

    strides = [8, 16, 32]

    total_time = 0.0
    num_samples = 0

    # 5. 추론 및 데이터 누적
    print("\nStarting inference on validation set...")
    with torch.no_grad():
        for images, targets in val_loader:
            # 리스트 형태의 텐서 패킹 처리 혹은 배치 텐서 이동
            if isinstance(images, list):
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)
                batch_size = images.size(0)
                
            # --- [시간 측정 시작] ---
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # 모델 예측 (FPN 특징 추출 -> Detection Head 통과)
            features = backbone(images)
            pred_hms, pred_regs = head(features)
            
            # --- [시간 측정 종료] ---
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # 배치 시간 누적
            total_time += (end_time - start_time)
            num_samples += batch_size
            
            all_detections = post_process(
                pred_hms, 
                pred_regs, 
                strides=strides, 
                threshold=0.05, 
                top_k=100
            )
            preds_metric = convert_to_metric_format(all_detections, device)
            
            # 4. 정답(targets) 데이터를 GPU로 이동
            targets_metric = []
            for t in targets:
                targets_metric.append({
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device)
                })
                
            # 5. 지표 업데이트
            metrics_val.update(preds_metric, targets_metric)

    print("\n" + "="*40)
    print("             EVALUATION REPORT          ")
    print("="*40)

    # 1) 평균 추론 시간 출력
    avg_inference_time = (total_time / num_samples) * 1000
    print(f"● Average Inference Time : {avg_inference_time:.2f} ms / image")

    # 2) mAP 지표 계산 및 출력
    results = metrics_val.compute()
    print(f"● mAP (@[0.5:0.95])      : {results['map'].item():.4f}")
    print(f"● mAP50                  : {results['map_50'].item():.4f}")
    print(f"● mAP75                  : {results['map_75'].item():.4f}")

    # 3) 객체별 AP50 출력
    id_to_cat = {0: 'pedestrian', 1: 'rider', 2: 'bike', 3: 'motor', 4: 'car', 
                5: 'bus', 6: 'truck', 7: 'traffic light', 8: 'traffic sign', 9: 'train'}

    print("\n● Per-Class AP50:")
    per_class_ap = results['map_per_class']
    for class_id, ap_value in enumerate(per_class_ap):
        class_name = id_to_cat.get(class_id, f"Class_{class_id}")
        if ap_value >= 0:
            print(f"  - {class_name:15s}: {ap_value.item():.4f}")
        else:
            print(f"  - {class_name:15s}: No Ground Truth")
    print("="*40)

    # # 6. 최종 지표 계산 및 결과 출력
    # print("\n" + "="*30)
    # print("       FINAL TEST RESULTS       ")
    # print("="*30)
    # 
    # # MAPCalculator의 최종 계산 메서드를 호출합니다.
    # # 만약 기존 val_loop 내부에서 다른 방식으로 출력하고 있다면 해당 객체의 동작에 맞게 호출 방식을 조정할 수 있습니다.
    # results = metrics_val.compute()
    # 
    # # 1) 전체 종합 mAP 출력
    # print(f"mAP (@[0.5:0.95]): {results['map'].item():.4f}")
    # print(f"mAP50:             {results['map_50'].item():.4f}")
    # print(f"mAP75:             {results['map_75'].item():.4f}")
# 
    # """# 2) 객체별(클래스별) AP 출력
    # # bdd_loader.py의 cat_to_id 역매핑 활용
    # id_to_cat = {0: 'pedestrian', 1: 'rider', 2: 'bike', 3: 'motor', 4: 'car', 
    #             5: 'bus', 6: 'truck', 7: 'traffic light', 8: 'traffic sign', 9: 'train'}
# 
    # print("\n--- Per-Class AP50 ---")
    # per_class_ap = results['map_per_class']  # 클래스별 AP 텐서 (일반적으로 IoU=0.5 기준)
    # for class_id, ap_value in enumerate(per_class_ap):
    #     class_name = id_to_cat.get(class_id, f"Class_{class_id}")
    #     # 데이터셋에 해당 클래스가 없어서 계산되지 않은 경우 -1이 들어있을 수 있습니다.
    #     if ap_value >= 0:
    #         print(f" - {class_name:15s}: {ap_value.item():.4f}")
    #     else:
    #         print(f" - {class_name:15s}: No Ground Truth or Not Evaluated")"""
    # 
    # if isinstance(results, dict):
    #     for metric_name, score in results.items():
    #         print(f"{metric_name}: {score}")
    # else:
    #     print("Evaluation finished. Please check the standard output or logs from MAPCalculator.")

if __name__ == "__main__":
    main()