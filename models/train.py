import torch
import sys

sys.path.append('../../preprocessed_dataset/')
sys.path.append('../../TransformerGrooveInfilling/model/')

from Subset_Creators.subsetters import GrooveMidiSubsetter
from dataset_loader import GrooveMidiDataset

from torch.utils.data import DataLoader
from transformer import GrooveTransformer
from io_layers import InputLayer,OutputLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filters = {"beat_type": ["beat"], "time_signature": ["4-4"], "master_id": ["drummer9/session1/9"]}

# LOAD SMALL TRAIN SUBSET
pickle_source_path = '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/Processed_On_17_05_2021_at_22_32_hrs'
subset_name = 'GrooveMIDI_processed_train'
metadata_csv_filename = 'metadata.csv'
hvo_pickle_filename = 'hvo_sequence_data.obj'

gmd_subsetter = GrooveMidiSubsetter(pickle_source_path=pickle_source_path, subset=subset_name,
                                    hvo_pickle_filename=hvo_pickle_filename, list_of_filter_dicts_for_subsets=[filters])

_, subset_list = gmd_subsetter.create_subsets()

subset_info = {"pickle_source_path": pickle_source_path, "subset": subset_name, "metadata_csv_filename":
    metadata_csv_filename, "hvo_pickle_filename": hvo_pickle_filename, "filters": filters}

mso_parameters = {"sr": 44100, "n_fft": 1024, "win_length": 1024, "hop_length": 441, "n_bins_per_octave": 16,
                  "n_octaves": 9, "f_min": 40, "mean_filter_size": 22}

voices_parameters = {"voice_idx": [2],  # closed hihat
                     "min_n_voices_to_remove": 1,
                     "max_n_voices_to_remove": 1,
                     "prob": [1],
                     "k": 1}

train_data = GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info, mso_parameters=mso_parameters,
                               max_aug_items=100, voices_parameters=voices_parameters,
                               sf_path="../../TransformerGrooveInfilling/soundfonts/filtered_soundfonts/", max_n_sf=1)

print("data len", train_data.__len__())

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)


# TRANSFORMER MODEL PARAMETERS
d_model = 128
nhead = 4
dim_feedforward = d_model * 10
dropout = 0.1
num_encoder_layers = 6
num_decoder_layers = 6
max_len = 32

embedding_size_in = 16
embedding_size_out = 27

TM = GrooveTransformer(d_model, nhead, dim_feedforward, dropout, num_encoder_layers,
                 num_decoder_layers, max_len).to(device)

IL_Encoder = InputLayer(embedding_size_in,d_model,dropout,max_len)
IL_Decoder = InputLayer(embedding_size_out,d_model,dropout,max_len)
OL = OutputLayer(embedding_size_out,d_model)

# TRAINING PARAMETERS
learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = torch.nn.CrossEntropyLoss()  # cambiar
optimizer = torch.optim.SGD(TM.parameters(), lr=learning_rate)  # cambiar


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y, idx) in enumerate(dataloader):
        print(X.shape, y.shape) # da Nx32xembedding_size
        X = X.permute(1,0,2)  # reorder dimensions to 32xNx embedding_size
        y = y.permute(1,0,2)  # reorder dimensions

        # Compute prediction and loss

        # y_shifted
        y_s = torch.zeros([1, y.shape[1], y.shape[2]])
        y_s = torch.cat((y_s, y[:-1,:,:]), dim=0)

        X = IL_Encoder(X)
        y_s = IL_Decoder(y_s)
        pred = model(X, y_s)
        pred = OL(pred)

        # TODO sum 3 different losses
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, TM, loss_fn, optimizer)
    print("Done!")
