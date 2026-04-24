import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork, RoIAlign

class DetectionAndFeatureProcessor(nn.Module):
    def __init__(self, num_classes, max_k=50, embed_dim=512):
        super(DetectionAndFeatureProcessor, self).__init__()
        self.max_k = max_k
        self.embed_dim = embed_dim
        
        # 1. Backbone: ResNet-50 (Pre-trained)
        # forward 시 매번 생성하지 않도록 __init__에서 한 번만 로드합니다.
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1 # Stride 4
        self.layer2 = resnet.layer2 # Stride 8
        self.layer3 = resnet.layer3 # Stride 16 (C4)
        self.layer4 = resnet.layer4 # Stride 32 (C5)
        
        # 2. FPN: P4(Stride 16), P5(Stride 32) 활용
        # ResNet-50 layer3: 1024ch, layer4: 2048ch
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[1024, 2048], 
            out_channels=256
        )
        
        # 3. 1-Stage Detector Head
        # P4 feature map (stride 16) 기준으로 예측 수행
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(256, 4, kernel_size=3, padding=1) # (dx, dy, dw, dh) 또는 (x, y, w, h)
        
        # 4. Milestone 2: Feature & Meta Processing
        # P4 stride가 16이므로 spatial_scale=1/16.0
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16.0, sampling_ratio=-1)
        
        # Feature compression (256 * 7 * 7 -> 496)
        self.feature_compressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 496)
        )
        
        # Meta embedding (Class Index(1) + Box(4) = 5 -> 16)
        self.meta_embedding = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        """
        x: (B, 3, H, W)
        Returns:
            mamba_tokens: (B, max_k, 512) - 2차 마일스톤 결과물
            det_outputs: (cls_logits, reg_preds) - 1차 마일스톤 학습용
        """
        batch_size = x.size(0)
        
        # --- [Step 1] Backbone Feature Extraction ---
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x) # (B, 1024, H/16, W/16)
        c4 = self.layer4(c3) # (B, 2048, H/32, W/32)
        
        # --- [Step 2] FPN ---
        # input dict의 key는 임의 지정 가능하나 in_channels_list 순서와 맞아야 함
        fpn_feats = self.fpn({"feat0": c3, "feat1": c4})
        p4 = fpn_feats["feat0"] # (B, 256, H/16, W/16)
        
        # --- [Step 3] Detection (Milestone 1) ---
        cls_logits = self.cls_head(p4)
        reg_preds = self.reg_head(p4)
        
        # --- [Step 4] Top-K Selection & ROI Extraction (Milestone 2) ---
        # 실제 학습 시에는 cls_logits에서 score가 높은 K개를 뽑아야 함
        # 현재는 로직 완성을 위해 간단한 가공 처리를 포함 (추후 NMS 대체 권장)
        with torch.no_grad():
            # 예시로 각 배치에서 임의의 K개 위치를 선정 (실제는 detector 결과 기반)
            # boxes: (B, K, 4) / classes: (B, K, 1)
            # 여기서는 dummy를 사용하되, 학습 시 실제 p4 feature와 연동됨
            dummy_boxes = torch.rand(batch_size, self.max_k, 4).to(x.device) * 100 
            dummy_classes = torch.randint(0, 10, (batch_size, self.max_k, 1)).float().to(x.device)

        # 1. ROI Align (P4 feature map 사용)
        rois = self._make_rois(dummy_boxes, batch_size) 
        roi_features = self.roi_align(p4, rois) # (B*K, 256, 7, 7)
        
        # 2. Feature Compression (B*K, 496)
        compressed_feats = self.feature_compressor(roi_features)
        
        # 3. Meta Embedding (B*K, 16)
        meta_info = torch.cat([dummy_classes, dummy_boxes], dim=-1) # (B, K, 5)
        meta_info = meta_info.view(-1, 5)
        embedded_meta = self.meta_embedding(meta_info)
        
        # 4. Final Fusion (B, K, 512)
        mamba_input_frame = torch.cat([compressed_feats, embedded_meta], dim=-1)
        mamba_input_frame = mamba_input_frame.view(batch_size, self.max_k, self.embed_dim)
        
        # 1차 마일스톤 학습을 위한 검출 결과와 2차 결과 토큰을 동시에 반환
        return mamba_input_frame, (cls_logits, reg_preds)

        return cls_logits, reg_preds

    def _make_rois(self, boxes, batch_size):
        # boxes shape: (B, K, 4) -> [x1, y1, x2, y2]
        # ROIAlign format: [batch_index, x1, y1, x2, y2]
        device = boxes.device
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(batch_size, self.max_k, 1)
        rois = torch.cat([batch_idx, boxes], dim=-1) # (B, K, 5)
        return rois.view(-1, 5).float()

# --- [3차 마일스톤 준비] Temporal Accumulation ---
def prepare_mamba_sequence(frames_list):
    """
    frames_list: 3~10개의 frame feature list. 각 (B, 50, 512)
    Returns: (B, Seq_Len, 512)
    """
    # 1. 각 프레임의 객체 토큰을 쌓음 (B, T, K, 512)
    sequence = torch.stack(frames_list, dim=1)
    
    # 2. Mamba 입력 형식을 위해 시퀀스 차원 평탄화 (B, T*K, 512)
    # 또는 객체별로 시간축을 태우는 등 전략에 따라 view 변경
    batch_size = sequence.size(0)
    sequence = sequence.view(batch_size, -1, 512) 
    
    return sequence


"""import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import feature_pyramid_network, RoIAlign

class DetectionAndFeatureProcessor(nn.Module):
    def __init__(self, num_classes, max_k=50, embed_dim=512):
        super(DetectionAndFeatureProcessor, self).__init__()
        self.max_k = max_k
        self.embed_dim = embed_dim
        
        # 1. Backbone: ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # P4 (layer3), P5 (layer4) 추출
        self.backbone = nn.ModuleDict({
            'layer3': resnet.layer3, # Stride 16
            'layer4': resnet.layer4  # Stride 32
        })
        
        # 2. FPN: P4, P5만 활용
        # ResNet-50의 layer3 채널은 1024, layer4는 2048
        self.fpn = feature_pyramid_network.FeaturePyramidNetwork(
            in_channels_list=[1024, 2048], 
            out_channels=256
        )
        
        # 3. 1-Stage Detector Head (Simple Version)
        # Class와 Box를 예측하는 간단한 구조
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(256, 4, kernel_size=3, padding=1) # x, y, w, h
        
        # 4. Milestone 2: Feature & Meta Processing
        # ROI Align: 7x7 해상도로 추출 후 Flatten
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16.0, sampling_ratio=-1)
        
        # Feature compression (7*7*256 -> 496)
        self.feature_compressor = nn.Sequential(
            nn.Linear(7 * 7 * 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 496)
        )
        
        # Meta embedding (Class Index(1) + Box(4) = 5 -> 16)
        self.meta_embedding = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        """
        x: (B, 3, H, W) - 단일 프레임 입력
        """
        batch_size = x.size(0)
        
        # Milestone 1: Detection
        # Backbone features
        c3 = self.backbone['layer3'](x) # Not used for FPN in this specific setup, but usually is.
        # Layer 1, 2 생략 후 바로 layer3, 4로 진행하는 구조는 
        # 실제로는 resnet 전체를 통과시켜야 하므로 아래와 같이 수정
        
        # 실제 구현시에는 resnet의 각 스테이지를 순차적으로 통과시킴
        # 여기서는 요약된 흐름만 표현
        features = {}
        out = resnet50(weights=ResNet50_Weights.DEFAULT).conv1(x)
        out = resnet50(weights=ResNet50_Weights.DEFAULT).bn1(out)
        out = resnet50(weights=ResNet50_Weights.DEFAULT).relu(out)
        out = resnet50(weights=ResNet50_Weights.DEFAULT).maxpool(out)
        out = resnet50(weights=ResNet50_Weights.DEFAULT).layer1(out)
        out = resnet50(weights=ResNet50_Weights.DEFAULT).layer2(out)
        c3 = resnet50(weights=ResNet50_Weights.DEFAULT).layer3(out) # 1024 ch
        c4 = resnet50(weights=ResNet50_Weights.DEFAULT).layer4(c3)  # 2048 ch
        
        # FPN (P4, P5)
        fpn_feats = self.fpn({"feat0": c3, "feat1": c4})
        p4, p5 = fpn_feats["feat0"], fpn_feats["feat1"]
        
        # Detector outputs (예시: p4 기준으로 검출 수행)
        cls_logits = self.cls_head(p4)
        reg_preds = self.reg_head(p4)
        
        # --- Top-K Selection (Simplify for Milestone 1) ---
        # 실제로는 NMS 등의 처리가 필요하지만, 2차 마일스톤을 위해 
        # 상위 K개의 고정된 텐서 크기를 확보하는 과정
        # (batch, K, 5) 형태의 boxes와 (batch, K, 1) 형태의 class_idx가 추출되었다고 가정
        
        # Dummy Top-K (학습 시에는 실제 검출 결과를 사용)
        # boxes: [batch, max_k, 4] (x, y, w, h)
        # scores: [batch, max_k, 1]
        dummy_boxes = torch.randn(batch_size, self.max_k, 4).to(x.device)
        dummy_classes = torch.randint(0, 10, (batch_size, self.max_k, 1)).float().to(x.device)
        
        # Milestone 2: Feature Compression
        # 1. ROI Align (p4 feature map에서 객체별 특징 추출)
        # rois format: [batch_idx, x1, y1, x2, y2]
        rois = self._make_rois(dummy_boxes, batch_size) 
        roi_features = self.roi_align(p4, rois) # (B*K, 256, 7, 7)
        
        # 2. Flatten & Compress
        roi_features = roi_features.view(roi_features.size(0), -1)
        compressed_feats = self.feature_compressor(roi_features) # (B*K, 496)
        
        # 3. Meta Embedding
        meta_info = torch.cat([dummy_classes, dummy_boxes], dim=-1) # (B, K, 5)
        meta_info = meta_info.view(-1, 5)
        embedded_meta = self.meta_embedding(meta_info) # (B*K, 16)
        
        # 4. Final Fusion (Mamba Input for 1 frame)
        mamba_input_frame = torch.cat([compressed_feats, embedded_meta], dim=-1) # (B*K, 512)
        mamba_input_frame = mamba_input_frame.view(batch_size, self.max_k, self.embed_dim)
        
        return mamba_input_frame # (Batch, 50, 512)

    def _make_rois(self, boxes, batch_size):
        # ROIAlign에 맞는 format [batch_index, x1, y1, x2, y2] 생성
        device = boxes.device
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).repeat(1, self.max_k, 1)
        rois = torch.cat([batch_idx, boxes], dim=-1)
        return rois.view(-1, 5)

# --- 2차 마일스톤: Temporal Accumulation (500ms / 10 frames) ---
def prepare_mamba_sequence(frames_data):
    """
    frames_data: List of 10 tensors, each (B, 50, 512)
    Returns: (B, 10, 50, 512) or Flattened Sequence for Mamba
    """
    # Mamba는 보통 (Batch, Seq_Len, Dim) 입력을 받으므로 
    # 객체 차원(50)과 시간 차원(10)을 어떻게 융합할지에 따라 달라짐
    # 가장 일반적인 방법은 각 객체의 시간 흐름을 보거나, 한 프레임 전체를 시퀀스로 보는 것임
    
    sequence = torch.stack(frames_data, dim=1) # (B, 10, 50, 512)
    return sequence"""