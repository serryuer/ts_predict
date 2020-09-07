import os, sys, logging, json, shutil, argparse
sys.path.append('/data03/yujunshuai/code/ts_predict/deepar')

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.net as net
from data.dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Eval')


def evaluate(model, test_loader, args, plot_num):
    model.eval()
    with torch.no_grad():
        all_loss = torch.zeros(1, device=args.device)
        all_labels = []
        all_predicts = []
        for i, (test_batch, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(args.device)
            #   id_batch = id_batch.unsqueeze(0).to(args.device)
            labels = labels.to(torch.float32).to(args.device)
            all_labels.append(labels)
            batch_size = test_batch.shape[1]
            predict_y = torch.zeros(batch_size, args.window_size, device=args.device) # scaled
            hidden = model.init_hidden(batch_size, args.device)
            cell = model.init_cell(batch_size, args.device)
            
            loss = torch.zeros(1, device=args.device)
            for t in range(args.window_size):
                loss_, y, hidden, cell = model(test_batch[t].unsqueeze(0), hidden, cell, labels)
                predict_y[:,t] = y
                loss += loss_
            all_predicts.append(predict_y)
            
            all_loss += loss.item() / args.window_size 
        
        all_labels = torch.cat(all_labels, axis=0)
        all_predicts = torch.cat(all_predicts, axis=0)
        summary_metric = {}
        summary_metric['loss'] = all_loss.detach().cpu().item() / len(test_loader)
        summary_metric['R2'] = 1 - (torch.nn.MSELoss(reduction='mean')(all_labels, all_predicts) / torch.std(labels)).detach().cpu().item()
        metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
        logger.info('- Full test metrics: ' + metrics_string)
    return summary_metric



def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                         predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                         alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})


        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()

if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name) 
    json_path = os.path.join(model_dir, 'args.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    args = utils.args(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    args.relative_metrics = args.relative_metrics
    args.sampling = args.sampling
    args.model_dir = model_dir
    args.plot_dir = os.path.join(model_dir, 'figures')
    
    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        args.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(args).cuda()
    else:
        args.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(args)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = TestDataset(data_dir, args.dataset, args.num_class)
    test_loader = DataLoader(test_set, batch_size=args.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    test_metrics = evaluate(model, loss_fn, test_loader, args, -1, args.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
