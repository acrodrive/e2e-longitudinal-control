import torch
import torch.nn as nn

class DetectionTransformerDecoder(nn.Module):
    """
    오브젝트 탐지를 위한 Transformer Decoder 모듈.
    학습 가능한 Object Queries를 내부에 포함하며, Pre-norm 구조를 권장함.
    """
    def __init__(self, num_queries=100, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # 1. Object Queries 정의: [num_queries, d_model]
        # 메인 모델에서 관리하지 않고 디코더 내부에서 관리하여 응집도를 높임
        self.object_queries = nn.Parameter(torch.randn(num_queries, d_model))
        #torch.randn으로 쿼리를 초기화하면 초기 학습이 매우 불안정할 수 있습니다. 나중에 정확도가 낮다면 nn.init.xavier_uniform_ 등을 사용하여 쿼리의 Magnitude를 조절하는 코드를 DetectionTransformerDecoder의 __init__에 추가하십시오.
        
        # 2. Decoder Layer 설정 (GELU 및 Pre-norm 권장)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu", # ReLU보다 성능 우수
            batch_first=True,
            norm_first=True    # 학습 안정성을 위한 Pre-norm
        )
        
        # 3. Transformer Decoder 정의
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

    def forward(self, encoder_hidden_states):
        """
        Args:
            encoder_hidden_states: 인코더에서 나온 시각 피처 [batch, seq_len, d_model]
        Returns:
            output: 디코더 출력 [batch, num_queries, d_model]
        """
        batch_size = encoder_hidden_states.shape[0]
        
        # Object Queries를 배치 사이즈만큼 복사: [batch, num_queries, d_model]
        query_embed = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transformer Decoder 실행
        # tgt: 학습 데이터(쿼리), memory: 인코더 출력값
        hs = self.decoder(tgt=query_embed, memory=encoder_hidden_states)
        
        return hs