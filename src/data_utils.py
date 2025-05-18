from torch.utils.data import Dataset
from torch import nn
import os
import numpy as np
import soundfile as sf
import torchaudio
import torch


class SceneFakeDataset(Dataset):
    def __init__(self, root_dir, target_length=64000):
        self.real_path = os.path.join(root_dir, 'real')
        self.fake_path = os.path.join(root_dir, 'fake')
        self.real_files = [os.path.join(self.real_path, f) for f in os.listdir(self.real_path)]
        self.fake_files = [os.path.join(self.fake_path, f) for f in os.listdir(self.fake_path)]
        self.all_files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        self.target_length = target_length
    def __len__(self):
        return len(self.all_files)
    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        waveform, sr = torchaudio.load(filepath)
        waveform = waveform[0]
        if waveform.size(0) < self.target_length:
            pad = self.target_length - waveform.size(0)
            waveform = nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:self.target_length]
        return waveform, self.labels[idx]


def extract_mfcc_mean(file_path, duration=4.0, sample_rate=16000, n_mfcc=40):
    waveform, sr = sf.read(file_path)
    waveform = waveform[:int(sample_rate * duration)]
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(torch.tensor(waveform), sr, sample_rate).numpy()
    waveform = torch.tensor(waveform).float().unsqueeze(0)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 64}
    )
    mfcc = mfcc_transform(waveform)
    mfcc_mean = mfcc.squeeze(0).mean(dim=1).numpy()
    return mfcc_mean


def load_or_extract_features(root_dir, cache_path):
    if os.path.exists(cache_path):
        print(f" Đang tải đặc trưng từ cache: {cache_path}")
        data = np.load(cache_path)
        return data['X'], data['y']
    print(f" Trích xuất đặc trưng MFCC từ {root_dir}")
    real_path = os.path.join(root_dir, 'real')
    fake_path = os.path.join(root_dir, 'fake')
    X, y = [], []
    for folder, label in [(real_path, 0), (fake_path, 1)]:
        for fname in os.listdir(folder):
            file_path = os.path.join(folder, fname)
            try:
                feat = extract_mfcc_mean(file_path)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f" Bỏ qua file: {file_path} ({str(e)})")
    X, y = np.array(X), np.array(y)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Đã lưu đặc trưng tại: {cache_path}")
    return X, y