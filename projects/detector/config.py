import torch

class Config:
    # --- 하드웨어 자원 최적화 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 32GB라면 640x640 기준 batch_size 16~32까지 노려볼 수 있습니다.
    # 하지만 안정적인 학습(Gradient 안정성)을 위해 16으로 시작하는 것을 권장합니다.
    batch_size = 16 
    num_workers = 8  # 5090의 속도를 맞추기 위해 데이터 로딩 스레드를 높게 잡습니다.
    pin_memory = True

    # --- 데이터 관련 (BDD100K 맞춤) ---
    img_size = 640  # 800 이상으로 키우면 정확도는 오르지만 속도가 급격히 느려집니다.
    num_classes = 10 
    
    # --- 모델 아키텍처 (32GB이기에 가능한 설정) ---
    num_queries = 100 # Detection 전용이므로 100~300 사이가 적당합니다.
    d_model = 256     # 512로 키울 수 있지만, 속도와 효율성 면에서 256이 검증된 수치입니다.
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048 # d_model*8 수준까지 올려도 메모리가 충분합니다.
    dropout = 0.1

    # --- 학습 최적화 ---
    lr = 1e-4
    lr_backbone = 1e-5
    weight_decay = 1e-4
    epochs = 50
    lr_drop = 40
    
    # Mixed Precision Training (FP16) 사용 여부
    # RTX 5090은 Tensor Core 성능이 압도적이므로 True 설정을 강력 권장합니다.
    use_amp = True 

    # 그래디언트 폭주 방지 (Transformer 필수)
    clip_max_norm = 0.1