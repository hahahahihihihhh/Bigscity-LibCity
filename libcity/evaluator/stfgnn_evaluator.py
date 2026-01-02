import pickle
import numpy as np
import os
import datetime

import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import json
from collections import defaultdict
from logging import getLogger
from libcity.utils import ensure_dir
from libcity.model import loss


def rse_np(preds, labels):
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    mse = np.sum(np.square(np.subtract(preds, labels)).astype('float32'))
    means = np.mean(labels)
    labels_mse = np.sum(np.square(np.subtract(labels, means)).astype('float32'))
    return np.sqrt(mse/labels_mse)


def mae_np(preds, labels):
    if isinstance(preds, np.ndarray):
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
    else:
        mae = np.abs(np.subtract(preds.cpu().numpy(), labels.cpu().numpy())).astype('float32')
    return np.mean(mae)


def rmse_np(preds, labels):
    mse = mse_np(preds, labels)
    return np.sqrt(mse)

def mse_np(preds, labels):
    if isinstance(preds, np.ndarray):
        return np.mean(np.square(np.subtract(preds, labels)).astype('float32'))
    else:
        return np.mean(np.square(np.subtract(preds.cpu().numpy(), labels.cpu().numpy())).astype('float32'))

def mape_np(preds, labels):
    if isinstance(preds, np.ndarray):
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
    else:
        mape = np.abs(np.divide(np.subtract(preds.cpu().numpy(), labels.cpu().numpy()).astype('float32'), labels.cpu().numpy()))
    return np.mean(mape)



def rae_np(preds, labels):
    mse = np.sum(np.abs(np.subtract(preds, labels)).astype('float32'))
    means = np.mean(labels)
    labels_mse = np.sum(np.abs(np.subtract(labels, means)).astype('float32'))
    return mse/labels_mse



def pcc_np(x, y):
    if not isinstance(x, np.ndarray):
        x, y = x.cpu().numpy(), y.cpu().numpy()
    x,y = x.reshape(-1),y.reshape(-1)
    return np.corrcoef(x,y)[0][1]


def node_pcc_np(x, y):
    if not isinstance(x, np.ndarray):
        x, y = x.cpu().numpy(), y.cpu().numpy()
    sigma_x = x.std(axis=0)
    sigma_y = y.std(axis=0)
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    cor = ((x - mean_x) * (y - mean_y)).mean(0) / (sigma_x * sigma_y + 0.000000000001)
    return cor.mean()

def corr_np(preds, labels):
    sigma_p = (preds).std(axis=0)
    sigma_g = (labels).std(axis=0)
    mean_p = preds.mean(axis=0)
    mean_g = labels.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((preds - mean_p) * (labels - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return correlation


def stemgnn_mape(preds,labels, axis=None):
    '''
    Mean absolute percentage error.
    :param labels: np.ndarray or int, ground truth.
    :param preds: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    mape = (np.abs(preds - labels) / (np.abs(labels)+1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


class STFGNNEvaluator(object):
    def __init__(self, config):
        self.metrics = config.get('metrics', ['MAE'])  # 评估指标, 是一个 list
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE',
                                'Accuracy', 'R2', 'EVAR']
        self.save_modes = config.get('save_mode', ['csv', 'json'])
        self.mode = config.get('evaluator_mode', 'single')  # or average
        self.mask_val = config.get('mask_val', None)
        self.config = config
        self.len_timeslots = 0
        self.result = {}  # 每一种指标的结果
        self.intermediate_result = {}  # 每一种指标每一个batch的结果
        self._check_config()
        self._logger = getLogger()

        # self.config = config
        # self.mask = self.config.get("mask", False)
        # self.out_catagory = "multi"
    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficStateEvaluator'.format(str(metric)))

    def _evaluate(self, output, groud_truth, mask = False, out_catagory = "multi"):
        """
        evluate the model performance
        : multi
        :param output: [n_samples, 12, n_nodes, n_features]
        :param groud_truth: [n_samples, 12, n_nodes, n_features]
        : single
        
        :return: dict [str -> float]
        """
        if out_catagory == 'multi':
            if bool(mask):
                if output.shape != groud_truth.shape:
                    groud_truth = np.expand_dims( groud_truth[...,0], axis=-1)
                assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
                batch, steps, scores, node = output.shape[0], output.shape[1], defaultdict(dict), output.shape[2]
                for step in range(steps):
                    y_pred = np.reshape(output[:,step],(batch, -1))
                    y_true = np.reshape(groud_truth[:,step],(batch,-1))
                    scores['masked_MAE'][f'horizon-{step}'] = masked_mae_np(y_pred, y_true, null_val=0.0)
                    scores['masked_RMSE'][f'horizon-{step}'] = masked_rmse_np(y_pred, y_true, null_val=0.0)
                    scores['masked_MAPE'][f'horizon-{step}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0
                    scores['node_wise_PCC'][f'horizon-{step}']= node_pcc_np(y_pred.swapaxes(1,-1).reshape((-1,node)), y_true.swapaxes(1,-1).reshape((-1,node)))
                    scores['PCC'][f'horizon-{step}'] = pcc_np(y_pred, y_true)
                scores['masked_MAE']['all'] = masked_mae_np(output,groud_truth ,null_val=0.0)
                scores['masked_RMSE']['all'] = masked_rmse_np( output,groud_truth, null_val=0.0)
                scores['masked_MAPE']['all'] = masked_mape_np( output,groud_truth, null_val=0.0) * 100.0
                scores['PCC']['all'] = pcc_np(output,groud_truth)
                scores["node_pcc"]['all'] = node_pcc_np(output, groud_truth)
            else:
                if output.shape != groud_truth.shape:
                    groud_truth = np.expand_dims( groud_truth[...,0], axis=-1)
                assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
                batch, steps, scores, node = output.shape[0], output.shape[1], defaultdict(dict), output.shape[2]
                for step in range(steps):
                    y_pred = output[:,step]
                    y_true = groud_truth[:,step]
                    scores['MAE'][f'horizon-{step}'] = mae_np(y_pred, y_true)
                    scores['RMSE'][f'horizon-{step}'] = rmse_np(y_pred, y_true)
                    # scores['MAPE'][f'horizon-{step}'] = mape_np(y_pred,y_true) * 100.0
                    scores['masked_MAPE'][f'horizon-{step}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0
                    scores['StemGNN_MAPE'][f'horizon-{step}'] = stemgnn_mape(y_pred, y_true) * 100.0
                    scores['PCC'][f'horizon-{step}'] = pcc_np(y_pred, y_true)
                    scores['node_wise_PCC'][f'horizon-{step}']= node_pcc_np(y_pred.swapaxes(1,-1).reshape((-1,node)), y_true.swapaxes(1,-1).reshape((-1,node)))
                scores['MAE']['all'] = mae_np(output,groud_truth)
                scores['RMSE']['all'] = rmse_np(output,groud_truth)
                scores['masked_MAPE']['all'] = masked_mape_np( output,groud_truth, null_val=0.0) * 100.0
                scores['StemGNN_MAPE']['all'] = stemgnn_mape(output,groud_truth) * 100.0
                scores['PCC']['all'] = pcc_np(output,groud_truth)
                scores['node_wise_PCC']['all'] = node_pcc_np(output.swapaxes(2,-1).reshape((-1,node)), groud_truth.swapaxes(2,-1).reshape((-1,node)))
        else:
            output = output.squeeze()
            groud_truth = groud_truth.squeeze()
            assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
            scores = defaultdict(dict)

            scores['RMSE']['all'] = rmse_np(output, groud_truth)
            scores['masked_MAPE']['all'] = masked_mape_np(output, groud_truth, null_val=0.0) * 100.0
            scores['PCC']['all'] = node_pcc_np(output, groud_truth)
            scores['rse']['all'] = rse_np(output, groud_truth)
            scores['rae']['all'] = rae_np(output, groud_truth)
            scores['MAPE']['all'] = stemgnn_mape(output, groud_truth) * 100.0
            scores['MAE']['all'] = mae_np(output, groud_truth)
            scores["node_pcc"]['all'] = node_pcc_np(output, groud_truth)
            scores['CORR']['all'] = corr_np(output, groud_truth)
        return scores


    def evaluate(self, output, groud_truth):
        if not isinstance(output, np.ndarray):
            output = output.cpu().numpy()
        if not isinstance(groud_truth, np.ndarray):
            groud_truth = groud_truth.cpu().numpy()
        return self._evaluate(output, groud_truth)

    def collect(self, batch):
        """
        收集一 batch 的评估输入

        Args:
            batch(dict): 输入数据，字典类型，包含两个Key:(y_true, y_pred):
                batch['y_true']: (num_samples/batch_size, timeslots, ..., feature_dim)
                batch['y_pred']: (num_samples/batch_size, timeslots, ..., feature_dim)
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']  # tensor
        y_pred = batch['y_pred']  # tensor
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        self.len_timeslots = y_true.shape[1]
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                if metric + '@' + str(i) not in self.intermediate_result:
                    self.intermediate_result[metric + '@' + str(i)] = []
        if self.mode.lower() == 'average':  # 前i个时间步的平均loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, :i], y_true[:, :i], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, :i], y_true[:, :i], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, :i], y_true[:, :i], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, :i], y_true[:, :i], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'Accuracy':
                        pass
                        # self.intermediate_result[metric + '@' + str(i)].append(
                        #     loss.accuracy_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.r2_score_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.explained_variance_score_torch(y_pred[:, :i], y_true[:, :i]).item())
        elif self.mode.lower() == 'single':  # 第i个时间步的loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'Accuracy':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.accuracy_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.r2_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.explained_variance_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
        else:
            raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))

    def evaluate_all_batch(self):
        """
        返回之前收集到的所有 batch 的评估结果
        """
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                self.result[metric + '@' + str(i)] = sum(self.intermediate_result[metric + '@' + str(i)]) / \
                                                     len(self.intermediate_result[metric + '@' + str(i)])
        return self.result

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        self._logger.info('Note that you select the {} mode to evaluate!'.format(self.mode))
        self.evaluate_all_batch()
        ensure_dir(save_path)
        if filename is None:  # 使用时间戳
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.json'.format(filename)))

        dataframe = {}
        if 'csv' in self.save_modes:
            for metric in self.metrics:
                dataframe[metric] = []
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    dataframe[metric].append(self.result[metric + '@' + str(i)])
            dataframe = pd.DataFrame(dataframe, index=range(1, self.len_timeslots + 1))
            dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.csv'.format(filename)))
            self._logger.info("\n" + str(dataframe))
        return dataframe

    def clear(self):
        """
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        self.result = {}
        self.intermediate_result = {}