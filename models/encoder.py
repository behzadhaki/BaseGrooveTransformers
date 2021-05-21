import torch


class Encoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm_encoder)

    def forward(self, src):
        print(src.shape)
        src = src.permute(1, 0, 2) # N, time_steps, 27 -> time_steps, N, 27
        print(src.shape)
        out = self.Encoder(src)
        print(out.shape)
        out = out.permute(1, 0, 2) # time_steps, N, 27 -> N, time_steps, 27
        print(out.shape)
        return out
