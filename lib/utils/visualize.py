import cv2
import torch
import numpy as np
from projects.CNN.models.loss import decode_to_bbox

def draw_on_video(video_path, model, output_path="output.mp4", threshold=0.5):
    """
    영상 파일에서 객체를 탐지하고 바운딩 박스를 그리는 추론 예시입니다.
    `pred_hm`이 임계값(threshold)을 넘는지 확인하는 방식은 제공된 답변을 참고했습니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 영상 정보 가져오기
    fps = int(cap.get(cv2.FPS))
    width = int(cap.get(cv2.FRAME_WIDTH))
    height = int(cap.get(cv2.FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model.eval()  # 모델 추론 모드 설정

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 모델 입력을 위한 전처리 (예시)
        img_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
        img_input = img_input.unsqueeze(0) # 배치를 위한 차원 추가

        # 모델 추론 (제공된 파일은 Loss 클래스이므로, 
        # 실제로는 이 Loss를 계산하기 위한 predictions 값을 주는 모델의 출력을 사용해야 합니다)
        with torch.no_grad():
            predictions = model(img_input) # 모델의 예측 결과 (예시) ###########################

        # 각 해상도 레벨(i=0, 1, 2)에 대해 반복
        for i in range(3):
            pred_hm, pred_reg = predictions[i] # 해당 레벨의 예측 결과 (제공된 파일 형식 참고)

            # 객체가 있을 확률(pred_hm)이 임계값보다 큰 위치를 찾습니다.
            # 이는 제공된 파일의 mask_bool 연산과 유사한 로직을 추론 단계에 적용한 것입니다.
            mask_bool = pred_hm.squeeze() > threshold 

            if mask_bool.any():
                # 객체가 발견된 그리드 위치 인덱스 (Batch, Y, X)
                batch_idx, y_idx, x_idx = torch.where(mask_bool)

                # 해당 위치의 예측 레그레션 값
                p_regs = pred_reg.permute(0, 2, 3, 1)[mask_bool] # [N, 4] -> [w, h, ox, oy]

                # 제공된 파일의 decode_to_bbox를 사용하여 바운딩 박스 좌표로 변환
                p_boxes = decode_to_bbox(p_regs, x_idx, y_idx)

                # 프레임에 바운딩 박스 그리기
                for box in p_boxes:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 초록색 박스

        out.write(frame) # 결과 프레임 저장

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

def draw_on_image(image_path, model, output_path="result.jpg", threshold=0.5):
    """
    단일 이미지 파일에서 객체를 탐지하고 바운딩 박스를 그립니다.
    """
    # 1. 이미지 로드
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not open image file {image_path}")
        return

    model.eval()

    # 2. 전처리 (모델 입력 규격에 맞게 변환)
    img_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
    img_input = img_input.unsqueeze(0) 

    # 3. 모델 추론
    with torch.no_grad():
        predictions = model(img_input)

    # 4. 결과 해석 및 그리기 (3개 레벨: s8, s16, s32)
    for i in range(3):
        pred_hm, pred_reg = predictions[i]
        
        # 임계값 이상의 객체 위치 마스킹
        mask_bool = pred_hm.squeeze() > threshold

        if mask_bool.any():
            batch_idx, y_idx, x_idx = torch.where(mask_bool)
            p_regs = pred_reg.permute(0, 2, 3, 1)[mask_bool]

            # 기존 loss.py의 로직을 활용한 좌표 복원
            # decode_to_bbox 함수는 이전 코드와 동일하게 사용
            p_boxes = decode_to_bbox(p_regs, x_idx, y_idx)

            for box in p_boxes:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                
                # 시각화 (초록색 박스 및 텍스트 추가)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Obj', (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 5. 결과 저장 및 확인
    cv2.imwrite(output_path, frame)
    print(f"Result saved to {output_path}")

    # (선택 사항) 로컬 환경이라면 화면에 바로 띄우기
    # cv2.imshow('Detection Result', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()