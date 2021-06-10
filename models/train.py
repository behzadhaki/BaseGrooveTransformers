import os
import torch
import wandb
import re
from models.transformer import GrooveTransformer


def calculate_loss(prediction, y, bce_fn, mse_fn):
    y_h, y_v, y_o = torch.split(y, int(y.shape[2] / 3), 2)  # split in voices
    pred_h, pred_v, pred_o = prediction

    bce_h = bce_fn(pred_h, y_h)  # batch, time steps, voices
    # bce_h_sum_voices = torch.sum(bce_h, dim=2)  # batch, time_steps
    # bce_hits = bce_h_sum_voices.mean()

    mse_v = mse_fn(pred_v, y_v)  # batch, time steps, voices
    # mse_v_sum_voices = torch.sum(mse_v, dim=2)  # batch, time_steps
    # mse_velocities = mse_v_sum_voices.mean()

    mse_o = mse_fn(pred_o, y_o)
    # mse_o_sum_voices = torch.sum(mse_o, dim=2)
    # mse_offsets = mse_o_sum_voices.mean()

    total_loss = bce_h + mse_v + mse_o

    _h = torch.sigmoid(pred_h)
    h = torch.where(_h > 0.5, 1, 0)  # batch=64, timesteps=32, n_voices=9

    h_flat = torch.reshape(h, (h.shape[0], -1))
    y_h_flat = torch.reshape(y_h, (y_h.shape[0], -1))
    n_hits = h_flat.shape[-1]
    hit_accuracy = (torch.eq(h_flat, y_h_flat).sum(axis=-1) / n_hits).mean()

    hit_perplexity = torch.exp(bce_h)

    # Log individual losses for hits, velocities and offsets
    individual_losses = {'hit_loss': bce_h, 'velocity_loss': mse_v, 'offset_loss': mse_o}
    wandb.log(individual_losses)

    return total_loss, hit_accuracy.item(), hit_perplexity.item()


def initialize_model(params):
    model_params = params["model"]
    training_params = params["training"]
    load_model = params["load_model"]

    groove_transformer = GrooveTransformer(model_params['d_model'], model_params['embedding_size_src'],
                                           model_params['embedding_size_tgt'], model_params['n_heads'],
                                           model_params['dim_feedforward'], model_params['dropout'],
                                           model_params['num_encoder_layers'], model_params['num_decoder_layers'],
                                           model_params['max_len'], model_params['device'])

    groove_transformer.to(model_params['device'])
    optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=training_params['learning_rate']) if \
        model_params['optimizer'] == 'adam' else torch.optim.SGD(groove_transformer.parameters(),
                                                                 lr=training_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, training_params['lr_scheduler_step_size'],
                                                gamma=training_params['lr_scheduler_gamma'])
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
                                       load_model['epoch']), run_path=load_model['dir'])
            checkpoint = torch.load(model_file.name)

        groove_transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']

    return groove_transformer, optimizer, scheduler, epoch


def train_loop(dataloader, groove_transformer, loss_fn, bce_fn, mse_fn, opt, scheduler, epoch, save, device):
    size = len(dataloader.dataset)
    groove_transformer.train()  # train mode
    loss = 0

    for batch, (x, y, idx) in enumerate(dataloader):

        x = x.to(device)
        y = y.to(device)

        # Compute prediction and loss
        # y_shifted
        y_s = torch.zeros([y.shape[0], 1, y.shape[2]]).to(device)
        y_s = torch.cat((y_s, y[:, :-1, :]), dim=1).to(device)

        pred = groove_transformer(x, y_s)
        loss, training_accuracy, training_perplexity = loss_fn(pred, y, bce_fn, mse_fn)

        # Backpropagation
        opt.zero_grad()
        loss.backward()

        # update optimizer and learning rate scheduler
        opt.step()
        scheduler.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print("hit accuracy:", training_accuracy)
            print("hit perplexity: ", training_perplexity)
            metrics = {'loss': loss, 'hit_accuracy': training_accuracy, 'hit_perplexity': training_perplexity}
            wandb.log(metrics)

    if save:
        # if we save the model in the wandb dir, it will be uploaded after training
        save_path = os.path.join(wandb.run.dir, "saved_models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_filename = os.path.join(save_path, "transformer_run_{}_Epoch_{}.Model".format(wandb.run.id, epoch))
        torch.save({'epoch': epoch, 'model_state_dict': groove_transformer.state_dict(),
                    'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss}, save_filename)
