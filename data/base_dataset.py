import torch.utils.data
import pandas as pd
import h5py
import numpy as np
from torch.nn import functional as F
import scipy.signal as signal


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.information = pd.read_csv(self.config['train_csv_path'])
        self.sensor_data = h5py.File(self.config['train_data_path'], 'r', driver='core')
        self.eq_event_names = self.information[self.information.trace_category == 'earthquake_local']['trace_name'].to_list()

    def __getitem__(self, index):
        eq_raw_data = np.array(self.sensor_data['earthquake']['local'][self.eq_event_names[index]])
        info = self.information[self.information.trace_name == self.eq_event_names[index]]

        # Compute input for NN
        noisy_stft = signal.stft(eq_raw_data.T, fs=100, window='hamming', nfft=1024, nperseg=512, noverlap=490,
                                 return_onesided=True, boundary=None, padded=False)[2]

        noisy_stft = np.concatenate((np.expand_dims(np.real(noisy_stft), 0),
                                     np.expand_dims(np.imag(noisy_stft), 0)), 0)

        if self.config['normalize_input']:
            noisy_stft = noisy_stft / 10.0

        noisy_stft = noisy_stft[:, :, :512, :]
        noisy_stft = torch.tensor(noisy_stft).float()
        noisy_stft = F.pad(noisy_stft, (0, 6, 0, 0), 'constant')

        # Compute label
        source_depth = float(info['source_depth_km'].iloc(0)[0])
        source_distance = info['source_distance_km'].iloc(0)[0]
        source_magnitude = info['source_magnitude'].iloc(0)[0]

        if self.config['normalize_labels']:
            source_depth = source_depth / 10.0
            source_distance = source_distance / 10.0
            source_magnitude = source_magnitude / 10.0

        return noisy_stft, source_depth, source_distance, source_magnitude

    def __len__(self):
        return len(self.eq_event_names)

    def __repr__(self):
        return self.__class__.__name__
