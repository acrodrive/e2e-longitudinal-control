import torch.nn as nn

class VisualTransformerEncoder(nn.Module):
    """
    CNN/Resampler 이후의 시퀀스 데이터를 처리하기 위한 Transformer Encoder.
    Pre-norm 구조와 GELU 활성화 함수를 기본으로 채택함.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # EncoderLayer를 별도 변수로 빼서 가독성 확보
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm 설정
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.transformer_encoder(x)

"""
self.encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=256, 
        nhead=8, 
        dim_feedforward=1024, # d_model 대비 적절한 크기로 조정
        dropout=0.1, 
        activation="gelu",    # ReLU보다 성능이 우수한 경우가 많음
        batch_first=True, 
        norm_first=True      # 학습 안정성을 위한 Pre-norm 설정
    ),
    num_layers=6
)
"""