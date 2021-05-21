import os

import torch
import sys

sys.path.append('../../preprocessed_dataset/')
sys.path.append('../../TransformerGrooveTap2Drum/model/')

import data_loader
from Subset_Creators.subsetters import GrooveMidiSubsetter

from torch.utils.data import DataLoader
from transformer import GrooveTransformer
from io_layers import InputLayer, OutputLayer

checkpoint_path = '../results/'
checkpoint_save_str = '../results/transformer_groove_tap2drum-epoch-{}'
epoch_save_div = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

print("data len", train_data.__len__(), '\n')

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

# TRANSFORMER MODEL PARAMETERS
d_model = 128
nhead = 8
dim_feedforward = d_model * 10
dropout = 0.1
num_encoder_layers = 1
num_decoder_layers = 1
max_len = 32

embedding_size_src = 27
embedding_size_tgt = 27

# TRAINING PARAMETERS
learning_rate = 1e-3
batch_size = 64

BCE = torch.nn.BCEWithLogitsLoss(reduction='sum')
MSE = torch.nn.MSELoss(reduction='sum')


def calculate_loss(prediction, y):
    div = int(y.shape[2] / 3)
    y_h, y_v, y_o = torch.split(y, div, 2)
    pred_h, pred_v, pred_o = prediction
    BCE_h = BCE(pred_h, y_h)
    MSE_v = MSE(pred_v, y_v)
    MSE_o = MSE(pred_o, y_o)
    print("BCE hits", BCE_h)
    print("MSE vels", MSE_v)
    print("MSE offs", MSE_o)
    return BCE_h + MSE_v + MSE_o


def load_model_from_latest_checkpoint():
    groove_transformer = GrooveTransformer(d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward,
                                           dropout, num_encoder_layers, num_decoder_layers, max_len).to(device)
    sgd_optimizer = torch.optim.SGD(groove_transformer.parameters(), lr=learning_rate)
    last_epoch = 0
    last_checkpoint = 0
    for root, dirs, files in os.walk(checkpoint_path):
        for name in files:
            checkpoint_epoch = int(name.split('-')[-1])
            last_checkpoint = checkpoint_epoch if checkpoint_epoch > last_checkpoint else last_checkpoint

    if last_checkpoint > 0:
        path = checkpoint_save_str.format(last_checkpoint)
        checkpoint = torch.load(path)
        groove_transformer.load_state_dict(checkpoint['model_state_dict'])
        sgd_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']

    return last_epoch, groove_transformer, sgd_optimizer


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    for batch, (X, y, idx) in enumerate(dataloader):
        optimizer.zero_grad()  # should be before calculating loss

        print(X.shape, y.shape)  # da Nx32xembedding_size
        X = X.permute(1, 0, 2)  # reorder dimensions to 32xNx embedding_size
        y = y.permute(1, 0, 2)  # reorder dimensions

        # Compute prediction and loss

        # y_shifted
        y_s = torch.zeros([1, y.shape[1], y.shape[2]])
        y_s = torch.cat((y_s, y[:-1, :, :]), dim=0)

        pred = model(X, y_s)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:  # ?
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if epoch % epoch_save_div == 0:
            checkpoint_save_path = checkpoint_save_str.format(str(epoch))
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, checkpoint_save_path)


if __name__ == "__main__":
    epoch, TM, optimizer = load_model_from_latest_checkpoint()
    while True:
        epoch += 1
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_dataloader, TM, calculate_loss, optimizer, epoch)
        print("Done!\n")
