import torch.nn as nn
from .encoder import VisualEncoder
from .decoder import ControlDecoder

class E2EControlModel(nn.Module):
    def __init__(self, embed_dim=512, state_dim=1):
        super().__init__()
        self.encoder = VisualEncoder(embed_dim)
        self.decoder = ControlDecoder(embed_dim, state_dim)

    def forward(self, img_seq, ego_state):
        memory = self.encoder(img_seq)
        pedal_val = self.decoder(memory, ego_state)
        return pedal_val