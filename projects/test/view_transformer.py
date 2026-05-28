import torch
import torch.nn as nn
import torch.nn.functional as F

class PureSpatialCrossAttention(nn.Module):
    def __init__(self, bev_h=640, bev_w=640, embed_dim=256):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        # BEV 그리드 좌표 미리 생성 (예: -64m ~ 64m)
        x = torch.linspace(-64, 64, bev_w)
        y = torch.linspace(-64, 64, bev_h)
        z = torch.linspace(-2, 4, 4) # Z축 4개 레벨 분할 예시
        
        # 3D 그리드 생성 (H, W, Z, 3)
        grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        self.register_buffer('grid_3d', grid.reshape(-1, 3)) # (N_points, 3)
        
        self.bev_embedding = nn.Parameter(torch.randn(bev_h, bev_w, embed_dim))
        self.proj_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, img_feats, intrinsic, extrinsic):
        # img_feats: (B, C, H_img, W_img) - 2D 이미지 피처
        # intrinsic: (B, 3, 3), extrinsic: (B, 4, 4) - Pitch, Roll 반영된 카메라 행렬
        
        B, C, H_i, W_i = img_feats.shape
        N_points = self.grid_3d.shape[0]
        
        # 1. 3D 점들을 Homogeneous 형태로 변환 (N_points, 4)
        ones = torch.ones(N_points, 1, device=img_feats.device)
        points_3d = torch.cat([self.grid_3d, ones], dim=-1).to(img_feats.device) # (N, 4)
        
        # 2. 월드/차량 좌표 -> 카메라 좌표 -> 2D 이미지 평면 투영 (정형 매트릭스 연산)
        # (이 과정에서 Batch별로 extrinsic, intrinsic을 연산하여 2D 픽셀 좌표 u, v 획득)
        # [구현 생략: Matrix Multiplication 및 카메라 평면 정규화]
        # 결과물 coords_2d: (B, N_points, 2) -> -1 ~ 1 사이로 정규화된 격자 좌표
        
        # 3. PyTorch 내장 grid_sample로 이미지 피처 맵에서 정보 추출
        # grid_sample 입력 포맷에 맞게 차원 변형 (B, 1, N_points, 2)
        sampling_grid = coords_2d.unsqueeze(1) 
        sampled_feat = F.grid_sample(img_feats, sampling_grid, align_corners=False) 
        # sampled_feat: (B, C, 1, N_points)
        
        # 4. 뜯어온 피처를 다시 BEV 구조(640, 640, Z)로 Reshape 후 융합
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1) # (B, N_points, C)
        bev_feat = sampled_feat.reshape(B, self.bev_h, self.bev_w, -1, C) # (B, H, W, Z, C)
        
        # Z축 정보를 채널로 합치거나 풀링하여 최종 BEV 맵 반환
        final_bev = bev_feat.mean(dim=3) # Z축 평균 예시 (프로젝션)
        
        return final_bev