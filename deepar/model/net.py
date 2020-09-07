'''Defines the neural network, loss function and metrics'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging

logger = logging.getLogger('DeepAR.Net')

class Net(nn.Module):
    def __init__(self, feature_dim, proj_dim, hidden_dim, num_layers_lstm, dropout=0.):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers_lstm = num_layers_lstm
        self.dropout = dropout

        super(Net, self).__init__()

        self.proj = nn.Linear(feature_dim, proj_dim)

        self.lstm = nn.LSTM(input_size=proj_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers_lstm,
                            bias=True,
                            batch_first=False,
                            dropout=dropout)
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()

        self.linear = nn.Linear(hidden_dim * self.num_layers_lstm, hidden_dim)
        self.act = nn.functional.tanh
        self.predict = nn.Linear(hidden_dim, 1)

        self.loss_fn = torch.nn.MSELoss() 

    def forward(self, x, hidden, cell, labels_batch):
        x = self.proj(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        output = F.relu(self.linear(hidden_permute))
        y = self.predict(output)
        return self.loss_fn(y, labels_batch), torch.squeeze(y), hidden, cell

    def init_hidden(self, input_size, device):
        return torch.zeros(self.num_layers_lstm, input_size, self.hidden_dim, device=device)

    def init_cell(self, input_size, device):
        return torch.zeros(self.num_layers_lstm, input_size, self.hidden_dim, device=device)



