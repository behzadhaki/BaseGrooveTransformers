import os
import torch
import sys

sys.path.append('../../preprocessed_dataset/')
sys.path.append('../../TransformerGrooveTap2Drum/model/')

import pandas as pd
import data_loader
from Subset_Creators.subsetters import GrooveMidiSubsetter

from torch.utils.data import DataLoader
from transformer import GrooveTransformer

checkpoint_path = '../results/'
checkpoint_save_str = '../results/transformer_groove_tap2drum-epoch-{}'

n_beats = 8

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

BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
MSE = torch.nn.MSELoss(reduction='none')


def calculate_loss(prediction, y, save_to_df=True, ep=0, save_path="."):
    div = int(y.shape[2] / 3)
    y_h, y_v, y_o = torch.split(y, div, 2)
    pred_h, pred_v, pred_o = prediction

    BCE_h = BCE(pred_h, y_h)  # 32, 3, 9 -> time steps, batch, voices
    BCE_h_mean_across_batches = torch.sum(torch.sum(BCE_h, dim=0), dim=1).mean()  # for back propagation

    MSE_v = MSE(pred_v, y_v)
    MSE_v_mean_across_batches = torch.sum(torch.sum(MSE_v, dim=0), dim=1).mean()

    MSE_o = MSE(pred_o, y_o)
    MSE_o_mean_across_batches = torch.sum(torch.sum(MSE_o, dim=0), dim=1).mean()

    if save_to_df:
        BCE_h_per_beat = torch.reshape(BCE_h, (n_beats, int(BCE_h.shape[0] / n_beats), BCE_h.shape[1],
                                               BCE_h.shape[2]))  # 8, 4, 3, 9 -> beats, steps in beat, batch, voice
        BCE_h_sum_beat = torch.sum(BCE_h_per_beat, dim=1)  # 8, 3, 9
        BCE_h_mean = torch.mean(BCE_h_sum_beat, dim=1)  # 8, 9 -> save this?
        df_hits = pd.DataFrame(BCE_h_mean.detach().numpy())
        df_hits.to_csv(save_path + 'epoch_' + str(ep) + '-BCE_hits.csv')

        MSE_v_per_beat = torch.reshape(MSE_v, (n_beats, int(MSE_v.shape[0] / n_beats), MSE_v.shape[1],
                                               MSE_v.shape[2]))  # 8, 4, 3, 9 -> beats, steps in beat, batch, voice
        MSE_v_sum_beat = torch.sum(MSE_v_per_beat, dim=1)  # 8, 3, 9
        MSE_v_mean = torch.mean(MSE_v_sum_beat, dim=1)  # 8, 9 -> save this?
        df_vels = pd.DataFrame(MSE_v_mean.detach().numpy())
        df_vels.to_csv(save_path + 'epoch_' + str(ep) + '-MSE_velocities.csv')

        MSE_o_per_beat = torch.reshape(MSE_o, (n_beats, int(MSE_o.shape[0] / n_beats), MSE_o.shape[1],
                                               MSE_o.shape[2]))  # 8, 4, 3, 9 -> beats, steps in beat, batch, voice
        MSE_o_sum_beat = torch.sum(MSE_o_per_beat, dim=1)  # 8, 3, 9
        MSE_o_mean = torch.mean(MSE_o_sum_beat, dim=1)  # 8, 9 -> save this?
        df_offsets = pd.DataFrame(MSE_o_mean.detach().numpy())
        df_offsets.to_csv(save_path + 'epoch_' + str(ep) + '-MSE_offsets.csv')

    return BCE_h_mean_across_batches + MSE_v_mean_across_batches + MSE_o_mean_across_batches


def load_model_from_latest_checkpoint(device):
    groove_transformer = GrooveTransformer(d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward,
                                           dropout, num_encoder_layers, num_decoder_layers, max_len, device).to(device)
    sgd_optimizer = torch.optim.SGD(groove_transformer.parameters(), lr=learning_rate)
    last_epoch = 0
    last_checkpoint = 0
    for root, dirs, files in os.walk(checkpoint_path):
        for name in files:
            if name.startswith('transformer_groove_tap2drum-epoch-'):
                checkpoint_epoch = int(name.split('-')[-1])
                last_checkpoint = checkpoint_epoch if checkpoint_epoch > last_checkpoint else last_checkpoint

    if last_checkpoint > 0:
        path = checkpoint_save_str.format(last_checkpoint)
        checkpoint = torch.load(path)
        groove_transformer.load_state_dict(checkpoint['model_state_dict'])
        sgd_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']

    return last_epoch, groove_transformer, sgd_optimizer


def train_loop(dataloader, model, loss_fn, optim, curr_epoch, epoch_save_div, df_path):
    size = len(dataloader.dataset)
    for batch, (X, y, idx) in enumerate(dataloader):
        save = (curr_epoch % epoch_save_div == 0)
        X = X.to(device)
        y = y.to(device)

        print(X.shape, y.shape)  # da Nx32xembedding_size

        # Compute prediction and loss

        # y_shifted
        y_s = torch.zeros([y.shape[0], 1, y.shape[2]]).to(device)
        y_s = torch.cat((y_s, y[:, :-1, :]), dim=1).to(device)

        pred = model(X, y_s)

        loss = loss_fn(pred, y, save_to_df=save, ep=curr_epoch, save_path=df_path)

        # Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch % 100 == 0:  # ?
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if save:
            checkpoint_save_path = checkpoint_save_str.format(str(curr_epoch))
            torch.save({'epoch': curr_epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, checkpoint_save_path)


if __name__ == "__main__":
    epoch, TM, optimizer = load_model_from_latest_checkpoint(device)
    epoch_save_div = 10
    df_path = "../results/losses_df/"
    while True:
        epoch += 1
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_dataloader, TM, calculate_loss, optimizer, epoch, epoch_save_div, df_path)
        print("Done!\n")
