import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence


class RNNFakeDetectionModel(nn.Module):
    def __init__(self, n_mfcc=13, hidden_size=64, dense_size=32, num_classes=2):
        super(RNNFakeDetectionModel, self).__init__()

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 400,
                'hop_length': 160,
                'n_mels': 40
            }
        )

        self.rnn = nn.RNN(input_size=n_mfcc, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, waveforms):
        mfcc_list = []
        for i in range(waveforms.size(0)):
            waveform = waveforms[i].unsqueeze(0)
            mfcc = self.mfcc_transform(waveform)
            mfcc = mfcc.squeeze(0).transpose(0, 1)
            mfcc_list.append(mfcc)

        mfcc_padded = pad_sequence(mfcc_list, batch_first=True)
        rnn_out, _ = self.rnn(mfcc_padded)
        last_hidden = rnn_out[:, -1, :]

        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
