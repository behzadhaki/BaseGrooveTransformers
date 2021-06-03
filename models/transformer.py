import torch

from encoder import Encoder
from decoder import Decoder
from io_layers import InputLayer,OutputLayer

from utils import get_tgt_mask


class GrooveTransformer(torch.nn.Module):
    def __init__(self, d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward, dropout,
                 num_encoder_layers, num_decoder_layers, max_len, device):
        super(GrooveTransformer, self).__init__()

        self.d_model = d_model
        self.embedding_size_src = embedding_size_src
        self.embedding_size_tgt = embedding_size_tgt
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.device = device

        self.InputLayerEncoder = InputLayer(embedding_size_src,d_model,dropout,max_len)
        self.Encoder = Encoder(d_model, nhead, dim_feedforward, dropout, num_encoder_layers)

        self.InputLayerDecoder = InputLayer(embedding_size_tgt,d_model,dropout,max_len)
        self.Decoder = Decoder(d_model, nhead, dim_feedforward, dropout, num_decoder_layers)
        self.OutputLayer = OutputLayer(embedding_size_tgt,d_model)

    def forward(self, src, tgt):
        # src Nx32xembedding_size_src
        # tgt Nx32xembedding_size_tgt
        mask = get_tgt_mask(self.max_len).to(self.device)

        x = self.InputLayerEncoder(src) # Nx32xd_model
        y = self.InputLayerDecoder(tgt) # Nx32xd_model
        memory = self.Encoder(x)        # Nx32xd_model
        out = self.Decoder(y, memory, tgt_mask=mask) #Nx32xd_model
        out = self.OutputLayer(out) #(Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,Nx32xembedding_size_src/3)

        return out


class GrooveTransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, embedding_size_src, embedding_size_mem, nhead, dim_feedforward, dropout,
                 num_encoder_layers, num_decoder_layers, max_len, device):
        super(GrooveTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding_size_src = embedding_size_src
        self.embedding_size_mem = embedding_size_mem
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_len = max_len
        self.device = device

        self.InputLayerEncoder = InputLayer(embedding_size_src,d_model,dropout,max_len)
        self.Encoder = Encoder(d_model, nhead, dim_feedforward, dropout, num_encoder_layers)
        self.OutputLayer = OutputLayer(embedding_size_mem,d_model)

    def forward(self, src):
        # src Nx32xembedding_size_src
        # tgt Nx32xembedding_size_tgt

        x = self.InputLayerEncoder(src) # Nx32xd_model
        memory = self.Encoder(x)        # Nx32xd_model
        out = self.OutputLayer(memory)  #(Nx32xembedding_size_mem/3,Nx32xembedding_size_mem/3,Nx32xembedding_size_mem/3)

        return out
