import torch
from utils import PositionalEncoding

class InputLayer(torch.nn.Module):
    def __init__(self, embedding_size,d_model,dropout,max_len):
        super(InputLayer,self).__init__()

        self.Linear = torch.nn.Linear(embedding_size,d_model, bias=True)
        self.ReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding(d_model, dropout, max_len)

    def forward(self,src):
        x = self.Linear(src)
        x = self.ReLU(x)
        out = self.PositionalEncoding(x)

        return out


class OutputLayer(torch.nn.Module):
    def __init__(self, embedding_size,d_model):
        super(OutputLayer,self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model,embedding_size,bias=True)


    def forward(self, decoder_out):
        y = self.Linear(decoder_out)
        y = torch.reshape(y, (decoder_out.shape[0], decoder_out.shape[1], 3, self.embedding_size // 3))

        _h = y[:,:,0,:]
        _v = y[:,:,1,:]
        _o = y[:,:,2,:]

        h=_h
        print(_v.shape)
        v = torch.sigmoid(_v)
        o = torch.tanh(_o)

        return h, v, o


