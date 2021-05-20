import torch
import sys

sys.path.append('../../preprocessed_dataset/')
sys.path.append('../../TransformerGrooveTap2Drum/model/')

import data_loader
from Subset_Creators.subsetters import GrooveMidiSubsetter

from torch.utils.data import DataLoader
from transformer import Transformer
from io_layers import InputLayer, OutputLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

####################################

filters = {"beat_type": ["beat"], "time_signature": ["4-4"], "master_id": ["drummer9/session1/9"]}

# LOAD SMALL TRAIN SUBSET
subset_info = {"pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2'
                                     '/Processed_On_17_05_2021_at_22_32_hrs',
               "subset": 'GrooveMIDI_processed_train',
               "metadata_csv_filename": 'metadata.csv',
               "hvo_pickle_filename": 'hvo_sequence_data.obj',
               "filters": filters
               }


gmd_subsetter = GrooveMidiSubsetter(pickle_source_path=subset_info["pickle_source_path"],
                                    subset=subset_info["subset"],
                                    hvo_pickle_filename=subset_info["hvo_pickle_filename"],
                                    list_of_filter_dicts_for_subsets=[filters])

_, subset_list = gmd_subsetter.create_subsets()
train_data = data_loader.GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info)

####################################

print("data len", train_data.__len__())

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

# TRANSFORMER MODEL PARAMETERS
d_model = 108
nhead = 3
dim_feedforward = d_model * 10
dropout = 0.1
num_encoder_layers = 6
num_decoder_layers = 6
max_len = 32

embedding_size_in = 27
embedding_size_out = 27

TM = Transformer(d_model, nhead, dim_feedforward, dropout, num_encoder_layers,
                 num_decoder_layers, max_len).to(device)

IL_Encoder = InputLayer(embedding_size_in, d_model, dropout, max_len)
IL_Decoder = InputLayer(embedding_size_out, d_model, dropout, max_len)
OL = OutputLayer(embedding_size_out, d_model)

# TRAINING PARAMETERS
learning_rate = 1e-3
batch_size = 64
epochs = 5

BCE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
optimizer = torch.optim.SGD(TM.parameters(), lr=learning_rate)  # cambiar

def calculate_loss(prediction, y):
    div = int(y.shape[2]/3)
    y_h, y_v, y_o = torch.split(y, div, 2)
    pred_h, pred_v, pred_o = prediction
    BCE_h = BCE(pred_h, y_h)
    MSE_v = MSE(pred_v, y_v)
    MSE_o = MSE(pred_o, y_o)
    return BCE_h + MSE_v + MSE_o

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y, idx) in enumerate(dataloader):
        print(X.shape, y.shape)  # da Nx32xembedding_size
        X = X.permute(1, 0, 2)  # reorder dimensions to 32xNx embedding_size
        y = y.permute(1, 0, 2)  # reorder dimensions

        # Compute prediction and loss

        # y_shifted
        y_s = torch.zeros([1, y.shape[1], y.shape[2]])
        y_s = torch.cat((y_s, y[:-1, :, :]), dim=0)

        X = IL_Encoder(X)
        y_s = IL_Decoder(y_s)
        pred = model(X, y_s)
        pred = OL(pred)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    epochs = 100000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, TM, calculate_loss, optimizer)
        print("Done!")
