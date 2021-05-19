import torch

from utils import PositionalEncoding,get_tgt_mask
from encoder import Encoder
from decoder import Decoder


class Transformer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers,
                 num_decoder_layers, max_len):
        super(Transformer, self).__init__()

        self.PositionalEncoder = PositionalEncoding(d_model, dropout, max_len)
        self.Encoder = Encoder(d_model, nhead, dim_feedforward, dropout, num_encoder_layers)
        self.Decoder = Decoder(d_model, nhead, dim_feedforward, dropout, num_decoder_layers)


    def forward(self, src=None, tgt=None, _mem=None, only_encoder=False, only_decoder=False):
        x = self.PositionalEncoder(src)
        memory = self.Encoder(x)
        mask = get_tgt_mask(tgt_len = tgt.shape[0])
        out = self.Decoder(tgt, memory, tgt_mask=mask)

        if only_encoder:
            return memory

        if only_decoder:
            return self.Decoder(tgt, src, tgt_mask=mask)

        return out