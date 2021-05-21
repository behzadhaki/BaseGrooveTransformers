import torch

d_model = 128
nhead = 4
dim_feedforward = d_model * 10
dropout = 0.1
num_encoder_layers = 6
num_decoder_layers = 6
max_len = 32
N = 64 # batch size

src_len = 32
tgt_len = 32

embedding_size = 27

# test input layer
from io_layers import InputLayer

src = torch.rand(src_len, N, embedding_size)

InputLayer = InputLayer(embedding_size, d_model, dropout, max_len)
y = InputLayer(src)
print(y.shape, y)

# test output layer
from io_layers import OutputLayer

OutputLayer = OutputLayer(embedding_size, d_model)
h, v, o = OutputLayer(y)
print(h, v, o)


# test transformer
from transformer import GrooveTransformer

embedding_size_src = 16
embedding_size_tgt = 27

src = torch.rand(src_len, N, embedding_size_src)
tgt = torch.rand(tgt_len, N, embedding_size_tgt)

TM = GrooveTransformer(d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward, dropout,
                       num_encoder_layers, num_decoder_layers, max_len)

print("TM")
h,v,o = TM(src,tgt)
print(h.shape, v.shape, o.shape)
print(h[0,0,:],v[0,0,:],o[0,0,:] )
