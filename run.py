"""
训练并评估单一模型的脚本
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from libcity.pipeline import run_model
from libcity.utils import ensure_dir
import pandas as pd
import numpy as np
import logging
import json


dataset_name, model_name = "SZ_TAXI", "DMKG_GNN"
log = "./logs/"
cache = "./libcity/cache/"
config_file = "config/{}/{}/config.json".format(dataset_name, model_name)


def train(config, total = 1):
    predict_steps, eval_metrics = config['output_window'], config['metrics']
    save_dir = os.path.join(log, config['dataset'], config['model'])

    # dir_name = "v1_2/"
    dir_name = ""
    if model_name == "DMKG_GNN":
        ke_dim, sparsity, n_layers = config['ke_dim'], config['sparsity'], config['n_layers']
        if dataset_name == "SZ_TAXI":
            dir_name += 'd{}_s{}'.format(ke_dim, sparsity)
        else:
            dir_name += 'd{}_s{}_l{}'.format(ke_dim, sparsity, n_layers)
        if config.get("without_spatial_enhanced_KG"):
            dir_name += "_without_spatial_enhanced_KG"
        if config.get("without_dynamic_meteorological_KG"):
            dir_name += "_without_dynamic_meteorological_KG"
        if config.get("without_spatial_enhanced"):
            dir_name += "_without_spatial_enhanced"
        if config.get("without_attribute_enhanced"):
            dir_name += "_without_attribute_enhanced"
        if config.get("without_state_propagation"):
            dir_name += "_without_state_propagation"
        if config.get("without_time_attention"):
            dir_name += "_without_time_attention"
    # if config.get('use_mh_adj', False):
    #     dir_name += '{}d_{}hop_s{}_b{}'.format(config['ke_dim'], config['max_hop'], config['sparsity'], config['batch_size'])
    save_dir = os.path.join(save_dir, dir_name)
    ensure_dir(save_dir)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log"),
        filemode='w'
    )
    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))
    # create logger
    logger = logging.getLogger()

    # 初始化
    exp_ids = []
    final_metrics = np.zeros([predict_steps, len(eval_metrics), total])

    # 独立训练total次
    for _i in range(total):
        # 解析参数
        config['exp_id'] = config['exp_id'] + _i
        exp_ids.append(config['exp_id'])
        other_args = {key: val for key, val in config.items() if key not in [
            'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
            val is not None}
        # 运行模型, 获取指标
        metrics = run_model(task=config['task'], model_name=config['model'], dataset_name=config['dataset'],
                 other_args=other_args)
        # 获取指标
        # result = pd.read_csv(cache + str(config['exp_id']) + '/' + 'evaluate_cache/' + modelName + '_' + datasetName + '.csv')
        final_metrics[:, :, _i] = metrics[eval_metrics].values

    logger.info('----------------------------------------------------------------------------')
    logger.info("Experiment ids: {}".format(exp_ids))

    # 计算指标均值和标准差
    avg_metrics = np.zeros([predict_steps, len(eval_metrics)])
    std_metrics = np.zeros([predict_steps, len(eval_metrics)])

    for _i in range(final_metrics.shape[0]):
        avg_metrics[_i] = np.mean(final_metrics[_i, :, :], axis=1)
        for id, metric in enumerate(eval_metrics):
            std_metrics[_i][id] = np.std(final_metrics[_i, :, :][id])

    # 保存各预测步指标均值和标准差
    df_avg_metrics = pd.DataFrame(avg_metrics, columns=eval_metrics)
    df_std_metrics = pd.DataFrame(std_metrics, columns=eval_metrics)
    df_avg_metrics.to_csv(os.path.join(save_dir, 'avg_result.csv'), index=False)
    df_std_metrics.to_csv(os.path.join(save_dir, 'std_result.csv'), index=False)
    logger.info('----------------------------------------------------------------------------')
    logger.info('Average: \n{} '.format(df_avg_metrics))
    logger.info('Standard deviation: \n{}'.format(df_std_metrics))

    # 保存所有预测步指标均值和标准差
    avg_steps_metrics = np.mean(avg_metrics, axis=0)
    std_steps_metrics = np.mean(std_metrics, axis=0)
    df_avg_steps_metrics = pd.DataFrame(np.array(avg_steps_metrics).reshape(1, len(eval_metrics)), columns=eval_metrics)
    df_std_steps_metrics = pd.DataFrame(np.array(std_steps_metrics).reshape(1, len(eval_metrics)), columns=eval_metrics)
    df_avg_steps_metrics.to_csv(os.path.join(save_dir, 'avg_steps_result.csv'), index=False)
    df_std_steps_metrics.to_csv(os.path.join(save_dir, 'std_steps_result.csv'), index=False)
    logger.info('----------------------------------------------------------------------------')
    logger.info('{} steps Average: \n{}'.format(predict_steps, df_avg_steps_metrics))
    logger.info('{} steps Standard deviation: \n{}'.format(predict_steps, df_std_steps_metrics))


if __name__ == '__main__':
    with open(config_file) as config:
        config = json.load(config)
        train(config)