import torch

class Decoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_decoder_layers):
        super(Decoder, self).__init__()
        norm_decoder = torch.nn.LayerNorm(d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm_decoder)

    def forward(self,tgt, memory, tgt_mask):
        return self.Decoder(tgt, memory, tgt_mask)