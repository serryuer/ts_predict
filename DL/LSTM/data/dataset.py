from __future__ import division
import numpy as np
import torch
import os, sys, json
import logging
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger('LSTM.Data')

def train_test_split(data_path, ratio):
    data = np.memmap(data_path, dtype='float64', mode='r')
    data = np.array(data.reshape((int(len(data)/46), 46)))
    paragraph_num = 0
    bad_row = []
    for i in range(data.shape[0]):
        if data[i, 0]==0:
            bad_row.append(i)
    train_test_split_index = bad_row[int(len(bad_row) * ratio)]
    return MyDataset(data[:train_test_split_index, :], bad_row), MyDataset(data[train_test_split_index:, :], bad_row)
    
class MyDataset(Dataset):
    def __init__(self, data, bad_row, max_seq_len = 100):
        self.data = data
        self.bad_row = bad_row
        self.train_len = self.data.shape[0]
        self.max_seq_len = max_seq_len
        logger.info(f'train_len: {self.train_len}')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        if index in self.bad_row:
            index += 1
        windows_start = 0
        for i in range(len(self.bad_row)):
            if self.bad_row[i] > index:
                break
        windows_start = self.bad_row[i - 1] if i != 0 else 0
        
        if index - windows_start > self.max_seq_len:
            x = self.data[index-self.max_seq_len+1:index+1, 2:-1]
            data_len = self.max_seq_len
        else:
            x = self.data[windows_start+1:index+1, 2:-1]
            x = np.concatenate([x, np.zeros([self.max_seq_len-(index-windows_start), x.shape[1]])])
            data_len = index - windows_start
        y = float(self.data[index][-1])
        return torch.tensor(x, dtype=torch.float32), torch.IntTensor([data_len]), torch.tensor([y * 10000], dtype=torch.float32)