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

