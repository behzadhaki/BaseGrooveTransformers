from transformer import Transformer
import torch

d_model = 16
nhead = 4
dim_feedforward = d_model*10
dropout = 0.1
num_encoder_layers = 6
num_decoder_layers = 6
max_len=32

N = 64
src_len = 32
tgt_len = 32

src = torch.rand(src_len, N, d_model)
tgt = torch.rand(tgt_len, N, d_model)

## test transformer
TM = Transformer(d_model, nhead, dim_feedforward, dropout, num_encoder_layers,
                 num_decoder_layers, max_len)

#print(TM.forward(src,tgt).shape,TM.forward(src,tgt))

## test input layer
from io_layers import InputLayer

embedding_size = 16
d_model = 128
InputLayer = InputLayer(embedding_size, d_model, dropout,max_len)
y = InputLayer(src)
print(y.shape, y)