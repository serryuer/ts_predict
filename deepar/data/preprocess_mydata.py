from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


data_path = 'data/data_104.dat'
save_path = 'data/deepar_data'


def prepare_data(data, window_size, stride_size, train_ratio):
    all_examples = []
    # all_labels = []
    window_start = 0
    input_size = window_size - stride_size
    max_end = (data[window_start:] == 0).argmax(0)[0] + window_start
    while window_start <= data.shape[0] - window_size - 1:
        if max_end - window_start < window_size:
            window_start = max_end + 1
            max_end = (data[window_start:] == 0).argmax(0)[0] + window_start
            continue
        example = np.zeros((window_size, 45), dtype='float32')
        example[:, 1:] = data[window_start:window_start + window_size, 2:]
        example[1:, 0] = data[window_start:window_start + window_size - 1, -1]
        # labels = np.zeros((window_size, 1), dtype='float32')
        # labels[:, -1] = data[window_start: window_start + window_size, -1]

        all_examples.append(example)
        # all_labels.append(labels)
        window_start += stride_size

    all_data = np.array(all_examples, dtype='float32')
    train_data_len = int(all_data.shape[0] * train_ratio)
    train_data = all_data[:train_data_len]
    test_data = all_data[train_data_len:]
    np.save(os.path.join(save_path, 'train_data'), train_data)
    np.save(os.path.join(save_path, 'test_data'), test_data)


if __name__ == '__main__':

    window_size = 192
    stride_size = 24


    array = np.memmap(data_path, dtype=np.float64)
    data_len = int(array.shape[0] / 46)
    data = np.memmap(data_path, dtype=np.float64, shape=(data_len, 46))
    prepare_data(data, window_size, stride_size, 0.9)
