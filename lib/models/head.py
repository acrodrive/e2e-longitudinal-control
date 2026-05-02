import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

    def _make_cls_head(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(), #inplace=True 고려?
            nn.Conv2d(256, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def _make_reg_head(self, in_ch, out_ch): # w, h, ox, oy
        return nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(), #inplace=True 고려?
            nn.Conv2d(256, out_ch, kernel_size=1)
        )

    def forward(self, p3, p4, p5):
        # 각 레벨의 출력 raw 값
        out_p3_raw = (self.cls_p3(p3), self.reg_p3(p3))
        out_p4_raw = (self.cls_p4(p4), self.reg_p4(p4))
        out_p5_raw = (self.cls_p5(p5), self.reg_p5(p5))

        processed_outs = []
        for cls_out, reg_out in [out_p3_raw, out_p4_raw, out_p5_raw]:
            # reg_out: [B, 4, H, W] -> (w, h, ox, oy)
            # w, h (채널 0, 1)에만 Softplus 또는 exp 적용하여 양수 보장
            # w_h = torch.exp(reg_out[:, :2, :, :]) # 음수 방지, 모델이 큰 수를 내보내는 경우 무한대에 가까운 값이 출력될 수 있다는 점을 인지해야 함
            w_h = F.softplus(reg_out[:, :2, :, :])
            w_h = torch.exp(torch.clamp(reg_out[:, :2, :, :], max=100))
            offset = reg_out[:, 2:, :, :]
            reg_out = torch.cat([w_h, offset], dim=1)
            processed_outs.append((cls_out, reg_out))
            
        # 2차 마일스톤 시 활성화: Mamba 입력 토큰 생성 (예: P4 기준)
        # mamba_token = self._generate_tokens(p4, boxes_gt, class_onehot_gt)
        # return [out_p3, out_p4, out_p5], mamba_token

        return processed_outs