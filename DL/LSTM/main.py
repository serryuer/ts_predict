import logging
# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)
import os
import argparse
import sys
sys.path.append('/data00/yujunshuai/code/ts_predict/DL/LSTM')
import torch
import torch.nn.functional as F
import torch.optim as optim
import dataset
from models.net import Net
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from transformers import AdamW
from trainer import Train
from data.dataset import train_test_split

from sklearn.metrics import *

# sys.argv.extend([
#     '-data_path=/data00/yujunshuai/code/ts_predict/data/data_289.dat',
#     '-lr=0.001',
#     '-epochs=100',
#     '-batch_size=64',
#     '-output_size=1',
#     '-save_dir=experients/test',
#     '-model_name=ts_predict'
# ])


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training settings
parser = argparse.ArgumentParser(description="Pytorch LSTM + Attention text classification")
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-dropout', type=float, default=0.3, help='the probability for dropout [default: 0.5]')
parser.add_argument('-batch_size',type=int,default=64,help='batch size for training [default: 128]')
parser.add_argument('-test_batch_size', type=int, default=128,help='input batch size for testing (default: 1024)')
parser.add_argument('-epochs', type=int, default=16, help='number of epochs for train [default: 20]')
parser.add_argument('-layer_size',type=int,default=1,help='the network layer [default 1]')
parser.add_argument('-bidirectional',type=bool,default=True,help='whether to use bidirectional network [default False]')
parser.add_argument('-hidden_size',type=int,default=256,help='number of hidden size for one rnn cell [default: 256]')
parser.add_argument('-embed_dim',type=int,default=43,help='number of embedding dimension [default: 128]')
parser.add_argument('-attention_size',type=int,default=12,help='attention size')

parser.add_argument('-random_seed',type=int,default=42,help='attention size')

# infomation output
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu,1 mean gpu [default: -1]')
parser.add_argument('-log_interval', type=int, default=1,help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-save_dir', type=str, default='experients', help='where to save the snapshot')
parser.add_argument('-model_name', type=str, default='ts_lstm', help='model name')
parser.add_argument('-early_stop_patience', type=int, default=10, help='early stop patience')

# model save/pre_train paths
parser.add_argument('-sequence_length',type=int,default=512,help='the length of input sentence [default 16]')
parser.add_argument('-output_size',type=int,default=1,help='number of classification [default: 2]')
parser.add_argument('-PAD',type=int,default=0,help='padding index')
parser.add_argument('-data_path',type=str,help='data path')

args = parser.parse_args()

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

print('Loading data...')
train_dataset, test_dataset = train_test_split(args.data_path, 0.9)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

args.cuda = args.device != -1 and torch.cuda.is_available()


print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    print('\t{}={}'.format(attr.upper(), value))

logging.info(f"train data all steps: {len(train_loader)}, " + 
                         f"test data all steps : {len(test_loader)}")

model = Net(args)
# model = DataParallel(model)
if args.cuda:
    model = model.cuda(args.device)

# Prepare  optimizer and schedule(linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
crit = torch.nn.CrossEntropyLoss()

trainer = Train(model_name=args.model_name,
                train_loader=train_loader,
                val_loader=test_loader,
                test_loader=None,
                model=model,
                optimizer=optimizer,
                epochs=args.epochs,
                print_step=args.log_interval,
                early_stop_patience=args.early_stop_patience,
                save_model_path=args.save_dir,
                save_model_every_epoch=False,
                metric=accuracy_score,
                num_class=args.output_size,
                tensorboard_path=args.save_dir,
                device=args.device)

print(trainer.train())