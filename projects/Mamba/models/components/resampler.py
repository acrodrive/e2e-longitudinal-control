import torch
import torch.nn as nn

class VisualResampler(nn.Module):
    def __init__(self, num_queries=128, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        
        # 1. 학습 가능한 쿼리 (Learnable Queries)
        # 트랜스포머 인코더로 들어갈 최종 토큰의 개수를 결정합니다.
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        # 2. 레벨 임베딩 (Level Embedding)
        # P4와 P5에서 온 토큰을 모델이 구분할 수 있게 합니다.
        self.level_embed_p4 = nn.Parameter(torch.randn(1, embed_dim))
        self.level_embed_p5 = nn.Parameter(torch.randn(1, embed_dim))
        
        # 3. Cross-Attention
        # Queries(Q)가 FPN 특징들(K, V)로부터 정보를 추출합니다.
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 4. Layer Norm & MLP (필수는 아니지만 안정성을 위해 추가)
        self.ln = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, p4, p5):
        # B: Batch, C: Channel, H, W: Height, Width
        B, C, H4, W4 = p4.shape
        _, _, H5, W5 = p5.shape
        
        # [Step 1] 텐서 펼치기 (Flatten)
        # (B, C, H, W) -> (B, H*W, C)
        feat_p4 = p4.flatten(2).transpose(1, 2)
        feat_p5 = p5.flatten(2).transpose(1, 2)
        
        # [Step 2] 레벨 임베딩 더하기
        # 모델이 "아 이건 P4꺼네" 하고 알게 해줍니다.
        feat_p4 = feat_p4 + self.level_embed_p4
        feat_p5 = feat_p5 + self.level_embed_p5
        
        # [Step 3] 모든 레벨의 특징 병합 (Total K, V Pool)
        # (B, 1450 + 375, 256) 형태로 합쳐집니다.
        combined_feats = torch.cat([feat_p4, feat_p5], dim=1)
        
        # [Step 4] Cross-Attention 수행
        # Q: [B, num_queries, embed_dim], K/V: [B, combined_len, embed_dim]
        # 쿼리를 배치 크기만큼 확장
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        # 핵심 연산: 수천 개의 픽셀을 num_queries개로 요약
        resampled_feat, _ = self.cross_attn(query=q, key=combined_feats, value=combined_feats)
        
        # [Step 5] Residual Connection & Feed Forward
        out = self.ln(resampled_feat + q)
        out = out + self.mlp(out)
        
        return out
