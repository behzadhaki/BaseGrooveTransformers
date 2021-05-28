import sys
from torch.utils.data import DataLoader
import os
import torch

sys.path.append('../../preprocessed_dataset/')

from transformer import GrooveTransformer
from Subset_Creators.subsetters import GrooveMidiSubsetter


def calculate_loss(prediction, y, bce_fn, mse_fn):
    div = int(y.shape[2] / 3)
    y_h, y_v, y_o = torch.split(y, div, 2)  # split in voices
    pred_h, pred_v, pred_o = prediction

    bce_h = bce_fn(pred_h, y_h)  # batch, time steps, voices
    bce_h_sum_voices = torch.sum(bce_h, dim=2)  # batch, time_steps
    bce_hits = bce_h_sum_voices.mean()
    print(bce_hits)

    _h = torch.sigmoid(pred_h)
    h = torch.where(_h > 0.5, 1, 0)

    print("after sigmoid", _h[:, :, 0])
    print("target hits", y_h[:, :, 0])
    print("prediction hits", h[:, :, 0])

    mse_v = mse_fn(pred_v, y_v)  # batch, time steps, voices
    mse_v_sum_voices = torch.sum(mse_v, dim=2)  # batch, time_steps
    mse_velocities = mse_v_sum_voices.mean()

    print("target velocities", y_v[:, :, 0])
    print("prediction velocities", pred_v[:, :, 0])

    mse_o = mse_fn(pred_o, y_o)
    mse_o_sum_voices = torch.sum(mse_o, dim=2)
    mse_offsets = mse_o_sum_voices.mean()

    print("target offsets", y_o[:, :, 0])
    print("prediction offsets", pred_o[:, :, 0])

    return bce_hits + mse_velocities + mse_offsets


def initialize_model(model_params, training_params, cp_info, load_from_checkpoint=False):
    groove_transformer = GrooveTransformer(model_params['d_model'], model_params['embedding_size_src'],
                                           model_params['embedding_size_tgt'], model_params['n_heads'],
                                           model_params['dim_feedforward'], model_params['dropout'],
                                           model_params['num_encoder_layers'], model_params['num_decoder_layers'],
                                           model_params['max_len'], model_params['device'])

    groove_transformer.to(model_params['device'])
    sgd_optimizer = torch.optim.SGD(groove_transformer.parameters(), lr=training_params['learning_rate'])
    epoch = 0

    if load_from_checkpoint:
        last_checkpoint = 0
        for root, dirs, files in os.walk(cp_info['checkpoint_path']):
            for name in files:
                if name.startswith('transformer'):
                    checkpoint_epoch = int(name.split('-')[-1])
                    last_checkpoint = checkpoint_epoch if checkpoint_epoch > last_checkpoint else last_checkpoint

        if last_checkpoint > 0:
            path = cp_info['checkpoint_save_str'].format(last_checkpoint)
            checkpoint = torch.load(path)
            groove_transformer.load_state_dict(checkpoint['model_state_dict'])
            sgd_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']

    return groove_transformer, sgd_optimizer, epoch


def train_loop(dataloader, groove_transformer, loss_fn, bce_fn, mse_fn, opt, epoch, save_epoch, cp_info, device):
    size = len(dataloader.dataset)
    for batch, (x, y, idx) in enumerate(dataloader):

        save = (epoch % save_epoch == 0)
        x = x.to(device)
        y = y.to(device)

        # print(x.shape, y.shape)  # N x time_steps x embedding_size

        # Compute prediction and loss
        # y_shifted
        y_s = torch.zeros([y.shape[0], 1, y.shape[2]]).to(device)
        y_s = torch.cat((y_s, y[:, :-1, :]), dim=1).to(device)

        pred = groove_transformer(x, y_s)
        loss = loss_fn(pred, y, bce_fn, mse_fn)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if save:
            if not os.path.exists(cp_info['checkpoint_path']):
                os.makedirs(cp_info['checkpoint_path'])
            checkpoint_save_path = cp_info['checkpoint_save_str'].format(str(epoch))
            torch.save({'epoch': epoch, 'model_state_dict': groove_transformer.state_dict(),
                        'optimizer_state_dict': opt.state_dict(), 'loss': loss}, checkpoint_save_path)


def load_dataset(Dataset, subset_info, filters, batch_sz, dataset_parameters={}):
    _, subset_list = GrooveMidiSubsetter(pickle_source_path=subset_info["pickle_source_path"],
                                         subset=subset_info["subset"],
                                         hvo_pickle_filename=subset_info["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[filters]).create_subsets()

    data = Dataset(subset=subset_list[0], subset_info=subset_info,**dataset_parameters)
    dataloader = DataLoader(data, batch_size=batch_sz, shuffle=True)

    return dataloader
