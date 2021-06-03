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

    def predict(self, src, use_thres = True, thres = 0.5, use_pd = False):
        with torch.no_grad():
            n_voices = self.embedding_size_tgt//3
            tgt = torch.zeros([src.shape[0], self.max_len, self.embedding_size_tgt]).to(self.device)

            for i in range(self.max_len):
                _h, v, o = self.forward(src, tgt) # Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,

                _h = torch.sigmoid(_h)

                if use_thres:
                    h = torch.where(_h > thres, 1, 0)

                if use_pd:
                    pd = torch.rand(_h.shape[0], _h.shape[1])
                    h = torch.where(_h > pd, 1, 0)

                tgt[:, i, 0: n_voices] = h[:,i,:]
                tgt[:, i, n_voices: 2 * n_voices ] = v[:,i,:]
                tgt[:, i, 2 * n_voices:] = o[:,i,:]

        return h,v,o


class GrooveTransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward, dropout,
                 num_encoder_layers, num_decoder_layers, max_len, device):
        super(GrooveTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding_size_src = embedding_size_src
        self.embedding_size_tgt = embedding_size_tgt
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
        self.OutputLayer = OutputLayer(embedding_size_tgt,d_model)

    def forward(self, src):
        # src Nx32xembedding_size_src

        x = self.InputLayerEncoder(src) # Nx32xd_model
        memory = self.Encoder(x)        # Nx32xd_model
        out = self.OutputLayer(memory)  #(Nx32xembedding_size_tgt/3,Nx32xembedding_size_tgt/3,Nx32xembedding_size_tgt/3)

        return out

    def predict(self, src, use_thres = True, thres = 0.5, use_pd = False):
        with torch.no_grad():
            n_voices = self.embedding_size_tgt//3
            tgt = torch.zeros([src.shape[0], self.max_len, self.embedding_size_tgt]).to(self.device)

            for i in range(self.max_len):
                _h, v, o = self.forward(src) # Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,Nx32xembedding_size_src/3,

                _h = torch.sigmoid(_h)

                if use_thres:
                    h = torch.where(_h > thres, 1, 0)

                if use_pd:
                    pd = torch.rand(_h.shape[0], _h.shape[1])
                    h = torch.where(_h > pd, 1, 0)

                tgt[:, i, 0: n_voices] = h[:,i,:]
                tgt[:, i, n_voices: 2 * n_voices ] = v[:,i,:]
                tgt[:, i, 2 * n_voices:] = o[:,i,:]

        return h,v,o
