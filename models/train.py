import os
import torch
import wandb
import re
import numpy as np
from models.transformer import GrooveTransformerEncoder, GrooveTransformer

h_div, v_div, o_div = -1, -1, -1


def calculate_loss(prediction, y, bce_fn, mse_fn, h_loss_multiplier, v_loss_multiplier, o_loss_multiplier):
    global h_div
    global v_div
    global o_div

    y_h, y_v, y_o = torch.split(y, int(y.shape[2] / 3), 2)  # split in voices
    pred_h, pred_v, pred_o = prediction

    bce_h = bce_fn(pred_h, y_h)  # batch, time steps, voices
    bce_h_sum_voices = torch.sum(bce_h, dim=2)  # batch, time_steps
    bce_hits = bce_h_sum_voices.mean()
    if h_div == -1:
        h_div = bce_hits.item()
    bce_hits = (bce_hits / h_div) * h_loss_multiplier

    mse_v = mse_fn(pred_v, y_v)  # batch, time steps, voices
    mse_v_sum_voices = torch.sum(mse_v, dim=2)  # batch, time_steps
    mse_velocities = mse_v_sum_voices.mean()
    if v_div == -1:
        v_div = mse_velocities.item()
    mse_velocities = (mse_velocities / v_div) * v_loss_multiplier

    mse_o = mse_fn(pred_o, y_o)
    mse_o_sum_voices = torch.sum(mse_o, dim=2)
    mse_offsets = mse_o_sum_voices.mean()
    if o_div == -1:
        o_div = mse_offsets.item()
    mse_offsets = (mse_offsets / o_div) * o_loss_multiplier

    total_loss = bce_hits + mse_velocities + mse_offsets

    _h = torch.sigmoid(pred_h)
    h = torch.where(_h > 0.5, 1, 0)  # batch=64, timesteps=32, n_voices=9

    h_flat = torch.reshape(h, (h.shape[0], -1))
    y_h_flat = torch.reshape(y_h, (y_h.shape[0], -1))
    n_hits = h_flat.shape[-1]
    hit_accuracy = (torch.eq(h_flat, y_h_flat).sum(axis=-1) / n_hits).mean()

    hit_perplexity = torch.exp(bce_hits)

    return total_loss, hit_accuracy.item(), hit_perplexity.item(), bce_hits, mse_velocities, mse_offsets


def initialize_model(params):
    model_params = params["model"]
    training_params = params["training"]
    load_model = params["load_model"]

    if model_params['encoder_only']:
        groove_transformer = GrooveTransformerEncoder(model_params['d_model'], model_params['embedding_size_src'],
                                                      model_params['embedding_size_tgt'], model_params['n_heads'],
                                                      model_params['dim_feedforward'], model_params['dropout'],
                                                      model_params['num_encoder_layers'],
                                                      model_params['num_decoder_layers'],
                                                      model_params['max_len'], model_params['device'])
    else:
        groove_transformer = GrooveTransformer(model_params['d_model'],
                                               model_params['embedding_size_src'],
                                               model_params['embedding_size_tgt'], model_params['n_heads'],
                                               model_params['dim_feedforward'], model_params['dropout'],
                                               model_params['num_encoder_layers'], model_params['num_decoder_layers'],
                                               model_params['max_len'], model_params['device'])

    groove_transformer.to(model_params['device'])
    optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=training_params['learning_rate']) if \
        model_params['optimizer'] == 'adam' else torch.optim.SGD(groove_transformer.parameters(),
                                                                 lr=training_params['learning_rate'])
    epoch = 0

    if load_model is not None:

        # If model was saved locally
        if load_model['location'] == 'local':

            last_checkpoint = 0
            # From the file pattern, get the file extension of the saved model (in case there are other files in dir)
            file_extension_pattern = re.compile(r'\w+')
            file_ext = file_extension_pattern.findall(load_model['file_pattern'])[-1]

            # Search for all continuous digits in the file name
            ckpt_pattern = re.compile(r'\d+')
            ckpt_filename = ""

            # Iterate through files in directory, find last checkpoint
            for root, dirs, files in os.walk(load_model['dir']):
                for name in files:
                    if name.endswith(file_ext):
                        checkpoint_epoch = int(ckpt_pattern.findall(name)[-1])
                        if checkpoint_epoch > last_checkpoint:
                            last_checkpoint = checkpoint_epoch
                            ckpt_filename = name

            # Load latest checkpoint found
            if last_checkpoint > 0:
                path = os.path.join(load_model['dir'], ckpt_filename)
                checkpoint = torch.load(path)

        # If restoring from wandb
        elif load_model['location'] == 'wandb':
            model_file = wandb.restore(load_model['file_pattern'].format(load_model['run'],
                                                                         load_model['epoch']),
                                       run_path=load_model['dir'])
            checkpoint = torch.load(model_file.name)

        groove_transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    return groove_transformer, optimizer, epoch


def train_loop(dataloader, groove_transformer, loss_fn, bce_fn, mse_fn, opt, epoch, save, device,
               encoder_only, test_inputs=None, test_gt=None, h_loss_mult=1, v_loss_mult=1, o_loss_mult=1):
    size = len(dataloader.dataset)
    groove_transformer.train()  # train mode
    loss = 0

    for batch, (x, y, idx) in enumerate(dataloader):

        opt.zero_grad()

        x = x.to(device)
        y = y.to(device)

        # Compute prediction and loss
        if encoder_only:
            pred = groove_transformer(x)
        else:
            # y_shifted
            y_s = torch.zeros([y.shape[0], 1, y.shape[2]]).to(device)
            y_s = torch.cat((y_s, y[:, :-1, :]), dim=1).to(device)
            pred = groove_transformer(x, y_s)

        loss, training_accuracy, training_perplexity, bce_h, mse_v, mse_o = loss_fn(pred, y, bce_fn, mse_fn,
                                                                                    h_loss_mult, v_loss_mult,
                                                                                    o_loss_mult)

        # Backpropagation
        loss.backward()

        # update optimizer
        opt.step()

        if batch % 1 == 0:
            wandb.log({'loss': loss.item(), 'hit_accuracy': training_accuracy, 'hit_perplexity': training_perplexity,
                       'hit_loss': bce_h, 'velocity_loss': mse_v, 'offset_loss': mse_o, 'epoch': epoch, 'batch': batch})
        if batch % 100 == 0:
            print('=======')
            current = batch * len(x)
            print(f"loss: {loss.item():>4f}  [{current:>4d}/{size:>4d}]")
            print("hit accuracy:", np.round(training_accuracy, 4))
            print("hit perplexity: ", np.round(training_perplexity, 4))
            print("hit bce: ", np.round(bce_h.item(), 4))
            print("velocity mse: ", np.round(mse_v.item(), 4))
            print("offset mse: ", np.round(mse_o.item(), 4))

    if save:
        # if we save the model in the wandb dir, it will be uploaded after training
        save_path = os.path.join(wandb.run.dir, "saved_models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_filename = os.path.join(save_path, "transformer_run_{}_Epoch_{}.Model".format(wandb.run.id, epoch))
        torch.save({'epoch': epoch, 'model_state_dict': groove_transformer.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'loss': loss}, save_filename)

    if test_inputs is not None and test_gt is not None:
        test_predictions_h, test_predictions_v, test_predictions_o = groove_transformer.predict(test_inputs,
                                                                                                use_thres=True,
                                                                                                thres=0.5)
        test_predictions = (test_predictions_h.float(), test_predictions_v.float(), test_predictions_o.float())
        test_loss, test_hits_accuracy, test_hits_perplexity, test_bce_h, test_mse_v, test_mse_o = \
            loss_fn(test_predictions, test_gt, bce_fn, mse_fn, h_loss_mult, v_loss_mult, o_loss_mult)
        wandb.log({'test_loss': test_loss.item(), 'test_hit_accuracy': test_hits_accuracy,
                   'test_hit_perplexity': test_hits_perplexity, 'test_hit_loss': test_bce_h.item(),
                   'test_velocity_loss': test_mse_v.item(), 'test_offset_loss': test_mse_o.item(), 'epoch': epoch})

    return loss.item()
