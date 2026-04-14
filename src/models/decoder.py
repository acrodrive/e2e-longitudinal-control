import torch
import torch.nn as nn

class ControlDecoder(nn.Module):
    def __init__(self, embed_dim=512, state_dim=1):
        super().__init__()
        self.query_embed = nn.Linear(state_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.output_head = nn.Linear(embed_dim, 1) # 최종 페달값 1개

    def forward(self, memory, ego_state):
        # memory (Encoder Hidden States): (B, Seq, embed_dim)
        # ego_state (Current Vehicle State): (B, state_dim)
        
        query = self.query_embed(ego_state).unsqueeze(1) # (B, 1, embed_dim)
        
        # Cross-Attention: Query=차량상태, K/V=이미지피처
        out = self.transformer_decoder(query, memory) # (B, 1, embed_dim)
        return self.output_head(out.squeeze(1))