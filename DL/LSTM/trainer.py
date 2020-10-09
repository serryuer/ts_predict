import logging
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import *
# log format
from tensorboardX import SummaryWriter
from tqdm import tqdm

class Train(object):
    def __init__(self, model_name, train_loader, val_loader, test_loader, model, optimizer, epochs, print_step,
                 early_stop_patience, save_model_path, num_class, save_model_every_epoch=False,
                 metric=f1_score, tensorboard_path=None, device=0):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.print_step = print_step
        self.early_stop_patience = early_stop_patience
        self.save_model_every_epoch = save_model_every_epoch
        self.save_model_path = save_model_path
        self.metric = metric
        self.num_class = num_class
        self.device = device

        self.tensorboard_path = tensorboard_path

        if not os.path.isdir(self.save_model_path):
            os.makedirs(self.save_model_path)
        if not os.path.isdir(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        self.best_val_epoch = 0
        self.best_val_score = 0

        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_path)

    def _save_model(self, model_name):
        torch.save(self.model, os.path.join(self.save_model_path, model_name + '.pt'))

    def _early_stop(self, epoch, score):
        if score > self.best_val_score:
            self.best_val_score = score
            self.best_val_epoch = epoch
            self._save_model('best-validate-model')
        else:
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + self.model_name + f"Validate has not promote {epoch - self.best_val_epoch}/{self.early_stop_patience}")
            if epoch - self.best_val_epoch > self.early_stop_patience:
                logging.info(self.model_name + f"-epoch {epoch}" + ":"
                             + f"Early Stop Train, best score locate on {self.best_val_epoch}, "
                             f"the best score is {self.best_val_score}")
                return True
        return False
    
    def _plot(self, epoch, true, predict, train):
        plt.scatter([i for i in range(true.shape[0])], true, c='lightblue')
        plt.plot([i for i in range(true.shape[0])], predict, color='red', linewidth=2)  
        plt.tight_layout()
        plt.savefig(f'{self.save_model_path}/{epoch}_{"train" if train else "test"}.png', dpi=300)
        

    def eval(self, epoch):
        logging.info(self.model_name + ":" + "## Start to evaluate. ##")
        self.model.eval()
        eval_loss = 0.0
        preds = None
        true_labels = None
        for batch_data in tqdm(self.val_loader, desc="Evaluating"):
            with torch.no_grad():
                batch_data = [data.cuda(self.device) for data in batch_data] if self.device != -1 else batch_data
                outputs = self.model(batch_data)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
                if preds is None:
                    preds = logits.detach().cpu().numpy()/10000
                    true_labels = batch_data[2].detach().cpu().numpy()/10000
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy()/10000, axis=0)
                    true_labels = np.append(true_labels, batch_data[2].detach().cpu().numpy()/10000, axis=0)
        result = {}
        from sklearn.metrics import r2_score, mean_squared_error
        result['R2'] = r2_score(true_labels, preds)
        result['mse'] = mean_squared_error(true_labels, preds)
        self._plot(epoch, true_labels, preds, False)
        return result

    def train(self):
        preds = None
        true_labels = None
        for epoch in range(self.epochs):
            tr_loss = 0.0
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + f"## The {epoch} Epoch, all {self.epochs} Epochs ! ##")
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + f"The current learning rate is {self.optimizer.param_groups[0].get('lr')}")
            self.model.train()
            since = time.time()
            for batch_count, batch_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                batch_data = [data.cuda(self.device) for data in batch_data] if self.device != -1 else batch_data
                outputs = self.model(batch_data)
                loss, logits = outputs[:2]
                loss = loss.sum()
                loss.backward()
                self.optimizer.step()
                tr_loss += loss.mean().item()
                if preds is None:
                    preds = logits.detach().cpu().numpy()/10000
                    true_labels = batch_data[2].detach().cpu().numpy()/10000
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy()/10000, axis=0)
                    true_labels = np.append(true_labels, batch_data[2].detach().cpu().numpy()/10000, axis=0)

                if (batch_count + 1) % self.print_step == 0:
                    logging.info(self.model_name + f"-epoch {epoch}" + ":"
                                 + f"batch {batch_count + 1} : loss is {tr_loss / (batch_count + 1)}, "
                                 + f"R2 is {r2_score(true_labels, preds)}"
                                 + f"mse is {mean_squared_error(true_labels, preds)}")
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_loss', tr_loss / (batch_count + 1),
                                              batch_count + epoch * len(self.train_loader))
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_mse',
                                              mean_squared_error(true_labels, preds),
                                              batch_count + epoch * len(self.train_loader))
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_r2',
                                              r2_score(true_labels, preds),
                                              batch_count + epoch * len(self.train_loader))

            self._plot(epoch, true_labels, preds, True)
            val_score = self.eval(epoch)
            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_r2', val_score['R2'], epoch)
            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_mse', val_score['mse'], epoch)

            logging.info(self.model_name + ": epoch" +
                         f"Epoch {epoch} Finished with time {format(time.time() - since)}, " +
                         f"validate accuracy score {val_score}")
            if self.save_model_every_epoch:
                self._save_model(f"{self.model_name}-{epoch}-{val_score['accuracy']}")
            if self._early_stop(epoch, val_score['R2']):
                break
        self.tb_writer.close()