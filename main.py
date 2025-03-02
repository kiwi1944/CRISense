# This is the code used for paper "Integrated Communication and Learned Recognizer with 
# Customized RIS Phases and Sensing Durations" in IEEE Transactions on Communications

# author: Huang Yixuan, email: huangyx@seu.edu.cn

# Public date: 2025-03-01


import os
import torch
import logging
import datetime
from trainer import Trainer
from config import get_config
from utils import seed_everything, prepare_dirs, save_config, renew_test_config_from_csv
from data_loader import get_test_loader, get_train_valid_loader
from channel_generation import channel_generation
import pandas as pd


def main(config):

    if config.is_train:
        used_channel = channel_generation(config) # generate point-to-point channels
        data_loader = get_train_valid_loader(batch_size=config.batch_size, valid_size=config.valid_size, train_data_scale=config.train_data_scale,
                                             shuffle=config.shuffle, num_workers=config.num_workers, ROI_num3=config.ROI_num3, data_name=config.data_name)
        trainer = Trainer(config, data_loader, used_channel)
        save_config(config)
        trainer.train()
    else: # require parameters: is_train, learned_start, wandb_data, test_index
        if len(config.test_wandb_data) == 0: raise RuntimeError("must input test_wandb_data when testing")
        wandb_data_csv = pd.read_csv(config.test_wandb_data) # test_wandb_data is downloaded from wandb webpage
        index = config.test_index - 1
        config = renew_test_config_from_csv(config, wandb_data_csv, index)
        used_channel = channel_generation(config)
        data_loader = get_test_loader(batch_size=config.batch_size, num_workers=config.num_workers, ROI_num3=config.ROI_num3, data_name=config.data_name)
        trainer = Trainer(config, data_loader, used_channel)
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config() # predefined configurations
    seed_everything(config.random_seed)

    config.time_str_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.ckpt_dir = config.ckpt_dir + config.time_str_now
    
    config.wandbflag = True if torch.cuda.is_available() and config.is_train else False # only use wandb when run on GPU
    wandb_project = config.wandb_project
    assert not (config.wandbflag == True and len(wandb_project) == 0), f'must input wandb_project when wandbflag is True'
    prepare_dirs(config) # ensure directories are setup
    if config.wandbflag: # log saving when trained on GPU
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=config.logs_dir + config.time_str_now + '.txt')
    elif not config.is_train: # log saving when test
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=config.test_wandb_data[0:-4] + '.txt')
        logging.info(f'************************* testing data row index: {config.test_index} *************************')
    else: # log outputting when debug
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {config.device}, using wandb: {config.wandbflag}, wandb project name: {wandb_project}')
    config.code_folder = os.path.split(os.getcwd())[1]

    if config.device.type == 'cpu':
        config.num_workers = 1
    
    main(config)
