import os, sys, logging, json, shutil, argparse
sys.path.append('/data03/yujunshuai/code/ts_predict/deepar')

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.net as net
from evaluate import evaluate
from data.dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='data/deepar_data/train_data.npy', help='train data path')
parser.add_argument('--val_data', default='data/deepar_data/test_data.npy', help='evaluate data path')
parser.add_argument('--model_dir', default='deepar/experiments/test_model', help='Directory containing args.json')
parser.add_argument('--use_gpu', default=False, help='whether to use gpu')
parser.add_argument('--gpu_num', default=0, help='whether to use gpu')
parser.add_argument('--feature_dim', default=44, help='feature dimension')
parser.add_argument('--proj_dim', default=45, help='porjection dimension')
parser.add_argument('--hidden_dim', default=256, help='hidden dimension of lstm')
parser.add_argument('--num_layers_lstm', default=2, help='layers of lstm')
parser.add_argument('--dropout', default=0, help='dropout rate of lstm')
parser.add_argument('--batch_size', default=16, help='batch size')
parser.add_argument('--window_size', default=192, help='window size')
parser.add_argument('--num_epochs', default=192, help='epoch nums')


parser.add_argument('--lr', default=16, help='learning rate')


def train(model: nn.Module,
          optimizer: optim,
          train_loader: DataLoader,
          test_loader: DataLoader,
          args,
          epoch: int) -> float:
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    for i, (train_batch, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(args.device)  # not scaled
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(args.device)  # not scaled

        loss = torch.zeros(1, device=args.device)
        hidden = model.init_hidden(batch_size, args.device)
        cell = model.init_cell(batch_size, args.device)

        for t in range(args.window_size):
            # if z_t is missing, replace it by output mu from the last time step
            loss_, y, hidden, cell = model(train_batch[t].unsqueeze_(0).clone(), hidden, cell, labels_batch[t])
            loss += loss_

        loss.backward()
        optimizer.step()
        loss = loss.item() / args.window_size  # loss per timestep
        loss_epoch[i] = loss
        # test_metrics = evaluate(model, test_loader, args, epoch)
        if i % 1000 == 0:
            test_metrics = evaluate(model, test_loader, args, epoch)
            model.train()
            logger.info(f'train_loss: {loss}')
        if i == 0:
            logger.info(f'train_loss: {loss}')
    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: optim,
                       args) -> None:
    logger.info('begin training and evaluation')
    best_test_R2 = float('inf')
    train_len = len(train_loader)
    loss_summary = np.zeros((train_len * args.num_epochs))
    R2_summary = np.zeros(args.num_epochs)
    for epoch in range(args.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, train_loader,
                                                                        test_loader, args, epoch)
        test_metrics = evaluate(model, test_loader, args, epoch)
        R2_summary[epoch] = test_metrics['R2']
        is_best = R2_summary[epoch] <= best_test_R2

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=is_best,
                              checkpoint=args.model_dir)

        if is_best:
            logger.info('- Found new best R2')
            best_test_R2 = R2_summary[epoch]
            best_json_path = os.path.join(args.model_dir, 'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best R2 is: %.5f' % best_test_R2)

        utils.plot_all_epoch(R2_summary[:epoch + 1], args.dataset + '_ND', args.plot_dir)
        utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len], args.dataset + '_loss', args.plot_dir)

        last_json_path = os.path.join(args.model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = args.model_dir

    if args.use_gpu:
        torch.cuda.manual_seed(42)
        logger.info('Using Cuda...')
        model = net.Net(args.feature_dim, args.proj_dim, args.hidden_dim, args.num_layers_lstm).cuda(args.gpu_num)
        args.device = torch.device(f'cuda:{args.gpu_num}')
    else:
        logger.info('Not using cuda...')
        model = net.Net(args.feature_dim, args.proj_dim, args.hidden_dim, args.num_layers_lstm)
        args.device = torch.device(f'cpu')

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logger.info('Loading the datasets...')
    train_set = MyTrainDataset(args.train_data)
    test_set = MyTestDataset(args.val_data)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(args.num_epochs))
    train_and_evaluate(model,
                       train_loader,
                       test_loader,
                       optimizer,
                       args)
