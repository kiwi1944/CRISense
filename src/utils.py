import os
import json
import logging
import numpy as np
import random
import torch


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prepare_dirs(config):
    if not os.path.exists(config.logs_dir) and config.wandbflag:
        os.makedirs(config.logs_dir)
    if not os.path.exists(config.ckpt_dir) and config.wandbflag:
        os.makedirs(config.ckpt_dir)


def save_config(config):
    if config.wandbflag:
        filename = config.time_str_now + '_params.json'
        param_path = os.path.join(config.logs_dir, filename)

        logging.info("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
        logging.info("[*] Param Path: {}".format(param_path))

        # data format process
        device = config.device
        config.device = str(config.device)

        with open(param_path, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)

        config.device = device


def renew_test_config_from_csv(config, wandb_data_csv, index): # set configs from wandb csv

    config.transmit_power = str(wandb_data_csv.loc[index, 'transmit_power'])
    config.noise_power = str(wandb_data_csv.loc[index, 'noise_power'])
    config.frequency = int(wandb_data_csv.loc[index, 'frequency'])
    config.distance = int(wandb_data_csv.loc[index, 'distance'])
    config.RIS_num2 = str(wandb_data_csv.loc[index, 'RIS_num2'])
    config.ROI_num3 = str(wandb_data_csv.loc[index, 'ROI_num3'])
    config.Rx = str(wandb_data_csv.loc[index, 'Rx'])
    config.Tx = str(wandb_data_csv.loc[index, 'Tx'])
    config.antenna_num = int(wandb_data_csv.loc[index, 'antenna_num'])
    config.self_interference = float(wandb_data_csv.loc[index, 'self_interference'])

    config.num_glimpses = int(wandb_data_csv.loc[index, 'num_glimpses'])
    config.rnn_hidden_size = int(wandb_data_csv.loc[index, 'rnn_hidden_size'])
    config.measure_embedding_hidden_size = str(wandb_data_csv.loc[index, 'measure_embedding_hidden_size'])
    config.RIS_phase_power_embedding_hidden_size = str(wandb_data_csv.loc[index, 'RIS_phase_power_embedding_hidden_size'])
    config.RIS_phase_customization_hidden_size = str(wandb_data_csv.loc[index, 'RIS_phase_customization_hidden_size'])
    config.classify_hidden_size = str(wandb_data_csv.loc[index, 'classify_hidden_size'])

    config.data_name = str(wandb_data_csv.loc[index, 'data_name'])
    config.learned_start = bool(wandb_data_csv.loc[index, 'learned_start'])
    config.random_seed = int(wandb_data_csv.loc[index, 'random_seed'])
    
    return config
