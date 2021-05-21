import torch

class Decoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_decoder_layers):
        super(Decoder, self).__init__()
        norm_decoder = torch.nn.LayerNorm(d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm_decoder)

    def forward(self, tgt, memory, tgt_mask):
        print(tgt.shape)
        tgt = tgt.permute(1, 0, 2)  # N, time_steps, 27 -> time_steps, N, 27
        print(tgt.shape)

        print(memory.shape)
        memory = memory.permute(1, 0, 2)  # N, time_steps, 27 -> time_steps, N, 27
        print(memory.shape)

        out = self.Decoder(tgt, memory, tgt_mask)

        print(out.shape)
        out = out.permute(1, 0, 2)  # time_steps, N, 27 -> N, time_steps, 27
        print(out.shape)

        return out
