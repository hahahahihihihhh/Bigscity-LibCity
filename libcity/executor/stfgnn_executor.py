import os
import time
import numpy as np
import torch
import math
import time
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
import tqdm
from torch.utils.tensorboard import SummaryWriter

from libcity.executor.abstract_executor import AbstractExecutor
# from executor.utils import get_train_loss
from libcity.model import loss
from libcity.utils.Optim import Optim
from libcity.utils.utils import ensure_dir
from functools import partial

# from model import loss
# from functools import partial

from libcity.utils import get_evaluator


def get_train_loss(train_loss):
    """
    get the loss func
    """
    if train_loss.lower() == 'none':
        print('Warning. Received none train loss func and will use the loss func defined in the model.')
        return None

    def func(preds, labels):

        if train_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif train_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif train_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif train_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif train_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif train_loss.lower() == 'huber':
            lf = loss.huber_loss
        elif train_loss.lower() == 'quantile':
            lf = loss.quantile_loss
        elif train_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif train_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif train_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif train_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif train_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif train_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        else:
            lf = loss.masked_mae_torch

        return lf(preds, labels)

    return func


class STFGNNExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.data_feature = data_feature
        self.config = config

        _device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(_device)
        self.model = model.to(self.device)
        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './libcity/cache/{}/'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._logger.info(self.model)

        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))

        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.train_loss = self.config.get("train_loss", "masked_mae")
        if self.train_loss == "none":
            self.train_loss = "masked_mae"
        self.criterion = get_train_loss(self.train_loss)
        self.cuda = self.config.get("cuda", True)
        self.best_val = 10000000
        self.optimizer = Optim(
            self.model.parameters(), self.config
        )
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('saved_model', True)
        self.epochs = self.config.get("max_epoch", 100)
        self.scaler = self.model.scaler
        self.num_batches = self.model.num_batches
        self.num_nodes = self.config.get("num_nodes", 0)
        self.batch_size = self.config.get("batch_size", 64)
        self.patience = self.config.get("patience", 20)
        self.lr_decay = self.config.get("lr_decay", False)
        self.mask = self.config.get("mask", True)
        self.output_dim = self.config.get('output_dim', 1)

    def train(self, train_data, valid_data):
        self._logger.info("begin training")
        wait = 0
        train_time, eval_time = [], []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = []
            start_time = time.time()
            for iter, batch in enumerate(train_data):
                batch.to_tensor(self.device)
                trainx = batch['X']  # [batch_size, window, num_nodes, dim]
                trainy = batch['y']  # [batch_size, horizon, num_nodes, dim]
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(self.scaler.inverse_transform(output),
                                      self.scaler.inverse_transform(trainy))
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            if self.lr_decay:
                self.optimizer.lr_scheduler.step()
            t1 = time.time()
            train_time.append(t1 - start_time)
            mtrain_loss = float(np.mean(train_loss))
            # valid_loss = []
            # y_preds = []
            # y_truths = []

            t2 = time.time()
            with torch.no_grad():
                self.model.eval()
                loss_func = self.model.calculate_loss
                losses = []
                for iter, batch in enumerate(valid_data):
                    batch.to_tensor(self.device)
                    loss = loss_func(batch)
                    self._logger.debug(loss.item())
                    losses.append(loss.item())
            end_time = time.time()
            eval_time.append(end_time - t2)
            mvalid_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mvalid_loss, epoch)
            # print(
            #     '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid mae {:5.4f}'.format(
            #         epoch, (time.time() - epoch_start_time), mtrain_loss, \
            #         mvalid_loss))

            if (epoch % self.log_every) == 0:
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, {:.2f}s'.\
                    format(epoch, self.epochs, mtrain_loss, mvalid_loss, (end_time - start_time))
                self._logger.info(message)

            if mvalid_loss < self.best_val:
                wait = 0
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '.format(self.best_val, mvalid_loss))
                self.best_val = mvalid_loss
                self.best_model = self.model
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch)
                    break

        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        self.model = self.best_model

    # def evaluate(self, test_data):
    #     exit(0)
    #     """
    #     use model to test data
    #
    #     Args:
    #         test_dataloader(torch.Dataloader): Dataloader
    #     """
    #     self._logger.info('Start evaluating ...')
    #     outputs = []
    #     realy = []
    #     seq_len = test_data.seq_len  #test_data["y_test"]
    #     self.model.eval()
    #     for iter, (x, y) in enumerate(test_data.get_iterator()):
    #         testx = torch.Tensor(x).to(self.device)
    #         testy = torch.Tensor(y).to(self.device)
    #         with torch.no_grad():
    #             # self.evaluator.clear()
    #             pred = self.model(testx)
    #             outputs.append(pred)
    #             realy.append(testy)
    #     realy = torch.cat(realy, dim=0)
    #     yhat = torch.cat(outputs, dim=0)
    #
    #     realy = realy[:seq_len, ...]
    #     yhat = yhat[:seq_len, ...]
    #
    #     realy = self.scaler.inverse_transform(realy)
    #     preds = self.scaler.inverse_transform(yhat)
    #
    #     res_scores = self.evaluator.evaluate(preds, realy)
    #     for _index in res_scores.keys():
    #         print(_index, " :")
    #         step_dict = res_scores[_index]
    #         for j, k in step_dict.items():
    #             print(j, " : ", k.item())

    def evaluate(self, test_data):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            for iter, batch in enumerate(test_data):
                batch.to_tensor(self.device)
                testx = batch['X']
                output = self.model.predict(batch)
                testy = batch['y']
                y_true = self.scaler.inverse_transform(testy)   # testy[..., :self.output_dim]
                y_pred = self.scaler.inverse_transform(output)   # output[..., :self.output_dim]
                y_truths.append(y_true.detach().cpu().numpy())
                y_preds.append(y_pred.detach().cpu().numpy())
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save(self.model.state_dict(), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
