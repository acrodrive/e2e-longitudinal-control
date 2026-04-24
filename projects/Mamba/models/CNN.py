import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork

class Detector1Stage(nn.Module):
    def __init__(self, num_classes):
        super(Detector1Stage, self).__init__()
        
        # 1. Backbone: ResNet-50
        # layer2(1/8), layer3(1/16), layer4(1/32) 추출
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.ModuleDict({
            'c3': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2),
            'c4': resnet.layer3,
            'c5': resnet.layer4
        })

        # 2. FPN: P4, P5 추출 (입력 채널: c4=1024, c5=2048)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[1024, 2048],
            out_channels=256,
            extra_blocks=None # 필요 시 P6, P7 추가 가능
        )

        # 3. Detection Head (P4, P5 각각 적용)
        # Output: Class Logits(num_classes) + Box(x, y, w, h) = num_classes + 4
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes + 4, kernel_size=1)
        )

    def forward(self, x):
        # Backbone feature extraction
        c3 = self.backbone['c3'](x)
        c4 = self.backbone['c4'](c3)
        c5 = self.backbone['c5'](c4)

        # FPN (P4: 1/16, P5: 1/32)
        # OrderedDict 형태로 입력
        features = self.fpn({"feat0": c4, "feat1": c5})
        p4, p5 = features["feat0"], features["feat1"]

        # Head application
        out_p4 = self.head(p4) # [B, num_classes+4, H/16, W/16] ##################### 이거 파일 분리 해달라고 하자
        out_p5 = self.head(p5) # [B, num_classes+4, H/32, W/32]

        return out_p4, out_p5

# 4. Loss Function (Simple Example)
def compute_loss(preds, targets, num_classes):
    """
    preds: List of [out_p4, out_p5]
    targets: 학습 데이터셋에 맞는 Ground Truth (포맷에 따라 할당 로직 필요)
    """
    total_loss = 0
    for pred in preds:
        # pred shape: [Batch, num_classes + 4, H, W]
        class_pred = pred[:, :num_classes, :, :]
        box_pred = pred[:, num_classes:, :, :]
        
        # 1-Stage 특성상 배경/객체 판단 및 정답 매칭 로직(예: Focal Loss + IoU Loss)이 
        # 여기에 추가되어야 합니다.
        
    return total_loss

# 테스트 실행
if __name__ == "__main__":
    num_classes = 10 # nuScenes 클래스 수 기준
    model = Detector1Stage(num_classes=num_classes)
    
    # 1280x720 입력 (Batch=1)
    sample_input = torch.randn(1, 3, 720, 1280)
    p4_out, p5_out = model(sample_input)

    print(f"P4 Output Shape: {p4_out.shape}") # [1, 14, 45, 80]
    print(f"P5 Output Shape: {p5_out.shape}") # [1, 14, 23, 40]
