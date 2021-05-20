import torch
from transformer import GrooveTransformer

import sys
sys.path.append('../../hvo_sequence/')


from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
import numpy as np


def predict(model, use_thres = True, thres = 0.5, use_pd = False):
    model.eval()
    with torch.no_grad():
        _h,v,o = TM(src,tgt)
        _h = _h.squeeze()
        v = v.squeeze()
        o = o.squeeze()
    
        _h = torch.sigmoid(_h)
    
        if use_thres:
            h = torch.where(_h > thres, 1, 0)
        
        if use_pd:
            pd = torch.rand(_h.shape[0], _h.shape[1])
            h = torch.where(_h > pd, 1, 0)
    
    return h,v,o

if __name__ == "__main__":

    d_model = 128
    nhead = 4
    dim_feedforward = d_model * 10
    dropout = 0.1
    num_encoder_layers = 6
    num_decoder_layers = 6
    max_len = 32
    N = 1  # batch size
    src_len = 32
    tgt_len = 32

    embedding_size_src = 16
    embedding_size_tgt = 27

    src = torch.rand(src_len, N, embedding_size_src)
    tgt = torch.rand(tgt_len, N, embedding_size_tgt)

    TM = GrooveTransformer(d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward, dropout,
                           num_encoder_layers, num_decoder_layers, max_len)


    h,v,o = predict(TM, use_pd=True)

    # convert to hvo
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)
    hvo_seq.add_time_signature(0, 4, 4, [4])
    hvo_seq.add_tempo(0, 50)

    hits = h
    vels = hits * v * 127
    offs = hits * o

    hvo_seq.hvo = np.concatenate((hits, vels, offs), axis=1)
    hvo_seq.save_audio(filename='./test.wav',sf_path = "../../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")
    #hvo_seq.to_html_plot(show_figure=True)


