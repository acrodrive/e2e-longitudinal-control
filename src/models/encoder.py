import torch
import torch.nn as nn
import torchvision.models as models

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        # x: (Batch, Seq, 3, 224, 224)
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        
        feat = self.feature_extractor(x) # (B*S, 512, 7, 7)
        feat = self.avgpool(feat).flatten(1) # (B*S, 512)
        feat = self.proj(feat) # (B*S, embed_dim)
        
        return feat.view(b, s, -1) # (B, Seq, embed_dim)