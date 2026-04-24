@torch.no_grad()
def inference(model, image_path, device, threshold=0.5):
    model.eval()
    image = cv2.imread(image_path)
    # 전처리...
    output = model(input_tensor.to(device))
    
    # 결과 필터링 (Confidence Threshold 적용)
    logits = output['pred_logits'].softmax(-1)[0, :, :-1]
    scores, labels = logits.max(-1)
    keep = scores > threshold
    
    boxes = output['pred_boxes'][0, keep]
    # 시각화 로직...