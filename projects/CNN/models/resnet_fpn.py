import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork

class ResNetFPN(nn.Module):
    def __init__(self, out_channels=256):
        super(ResNetFPN, self).__init__()
        # Backbone: ResNet-50 (Fine-tuning 모드)
        res50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 가벼운 연산을 위해 초기 layer들 고정(Freezing) 고려 가능
        self.body = nn.ModuleDict({
            'c3': nn.Sequential(res50.conv1, res50.bn1, res50.relu, res50.maxpool, res50.layer1, res50.layer2),
            'c4': res50.layer3,
            'c5': res50.layer4
        })

        # FPN (P3, P4, P5)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048], 
            out_channels=out_channels
        )

    def forward(self, x):
        c3 = self.body['c3'](x)
        c4 = self.body['c4'](c3)
        c5 = self.body['c5'](c4)
        
        fpn_inputs = {'0': c3, '1': c4, '2': c5}
        fpn_outs = self.fpn(fpn_inputs)
        
        return fpn_outs['0'], fpn_outs['1'], fpn_outs['2']