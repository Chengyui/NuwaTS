

import torch.utils.data as torch_data
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def time_continues(timestamps):
    fixed_interval = timestamps[1] - timestamps[0]  # calculate the interval

    is_continuous = True

    for i in range(1, len(timestamps)):
        interval = timestamps[i] - timestamps[i - 1]
        if interval != fixed_interval:
            is_continuous = False
            return is_continuous
            break
    return is_continuous


def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data), min=np.min(data))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data), std=np.std(data))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        # std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastGrid(torch_data.Dataset):
    def __init__(self, data, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1,
                 df_length=1000, channel_num=1000, channel_group=1, origin_missrate=0):
        self.window_size = window_size  # 12
        self.interval = interval  # 1
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        self.data = data
        if len(data.shape) > 2:
            self.data = data.reshape((data.shape[0], -1))
        self.df_length = self.data.shape[0]
        self.channel_num = self.data.shape[1]
        self.channel_group = channel_group

        self.data = self.data[:, :self.channel_num - (self.channel_num % channel_group)]
        self.data = self.data.reshape((self.df_length, channel_group, -1))
        self.data = self.data.transpose(2, 0, 1)
        self.data = self.data.reshape((-1, channel_group))


        self.x_end_idx = self.get_x_end_idx()
        self.scaler = StandardScaler()
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.data_origin_mask = torch.rand(self.data.shape)
        self.data_origin_mask[self.data_origin_mask > origin_missrate] = 1
        self.data_origin_mask[self.data_origin_mask <= origin_missrate] = 0

    def __getitem__(self, index):
        hi = self.x_end_idx[index]  # 12
        lo = hi - self.window_size  # 0
        train_data = self.data[lo: hi].copy()  # 0:12
        target_data = self.data[hi:hi + self.horizon].copy()  # 12:24

        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        seq_x_mark = self.data_origin_mask[lo:hi]
        seq_y_mark = torch.zeros((y.shape[0], 1))
        return x, y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):

        x_index_set = []
        for channel in range(self.channel_num // self.channel_group):
            # x_index_set.append(range(channel*self.df_length + self.window_size, (channel+1)*self.df_length - self.horizon + 1))
            x_index_set.extend(list(
                range(channel * self.df_length + self.window_size, (channel + 1) * self.df_length - self.horizon + 1)))
        # x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

