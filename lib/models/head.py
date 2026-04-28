import torch
import torch.nn as nn
import math

class DetectionHead(nn.Module):
    def __init__(self, num_classes, fpn_channels=256):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes

        self.cls_p3 = self._make_cls_head(fpn_channels, num_classes)
        self.cls_p4 = self._make_cls_head(fpn_channels, num_classes)
        self.cls_p5 = self._make_cls_head(fpn_channels, num_classes)

        self.reg_p3 = self._make_reg_head(fpn_channels, 4)
        self.reg_p4 = self._make_reg_head(fpn_channels, 4)
        self.reg_p5 = self._make_reg_head(fpn_channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        prob = 0.01
        bias_value = -math.log((1 - prob) / prob)

        nn.init.constant_(self.cls_p3[-2].bias, bias_value)
        nn.init.constant_(self.cls_p4[-2].bias, bias_value)
        nn.init.constant_(self.cls_p5[-2].bias, bias_value)

        """
        # 2차 마일스톤: Mamba Tokenizer 관련 (나중에 활성화)
        # self.roi_align = ...
        # self.mlp_box = ...
        # self.mlp_feat = ...
        """

    def _make_cls_head(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(), #inplace=True 고려?
            nn.Conv2d(256, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def _make_reg_head(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(), #inplace=True 고려?
            nn.Conv2d(256, out_ch, kernel_size=1)
        )

    def forward(self, p3, p4, p5, boxes_gt=None, class_onehot_gt=None):
        # 1차 마일스톤: 독립 헤드를 통한 예측
        out_p3 = (self.cls_p3(p3), self.reg_p3(p3))
        out_p4 = (self.cls_p4(p4), self.reg_p4(p4))
        out_p5 = (self.cls_p5(p5), self.reg_p5(p5))

        """
        # 2차 마일스톤 시 활성화: Mamba 입력 토큰 생성 (예: P4 기준)
        # mamba_token = self._generate_tokens(p4, boxes_gt, class_onehot_gt)
        # return [out_p3, out_p4, out_p5], mamba_token
        """
        
        return [out_p3, out_p4, out_p5]