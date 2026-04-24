import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from .components.resampler import VisualResampler
from .components.visual_encoder import VisualTransformerEncoder
from .components.detection_decoder import DetectionTransformerDecoder
from .components.heads import DetectionHead

class ObjectDetector(nn.Module):
    def __init__(self, config):
        super().__init__()

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        backbone = model.backbone

        self.backbone = backbone     

        self.resampler = VisualResampler(
            num_queries=128,
            embed_dim=256,
            num_heads=8
            )

        self.encoder = VisualTransformerEncoder(
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
            )

        self.decoder = DetectionTransformerDecoder(
            num_queries=config.num_queries,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1
            )
        
        self.heads = DetectionHead(
            d_model=256,
            num_classes=config.num_classes
            )

    def forward(self, x):
        # x: [Batch, 3, w, h] (이미지)
        
        # CNN -> FPN 특징 추출
        features = self.backbone(x)
        p4, p5 = features['2'], features['3']
        
        # Resampler: 수만 개의 픽셀을 128개로 압축 (여기서 OOM 방어!)
        latent_tokens = self.resampler(p4, p5)
        
        # Encoder: 128개 토큰 간의 관계 파악
        memory = self.encoder(latent_tokens)
        
        hs = self.decoder(memory)
        
        outputs = self.heads(hs)
        
        return outputs