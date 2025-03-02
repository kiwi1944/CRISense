import argparse


arg_lists = []
parser = argparse.ArgumentParser(description='RAM')

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# following parameters are used for dataset loading
system_arg = add_argument_group('imaging system Params')
system_arg.add_argument('--transmit_power', type=str, default='70', help='dBm')
system_arg.add_argument('--noise_power', type=str, default='', help='dBm')
system_arg.add_argument('--frequency', type=int, default=3, help='working frequency, unit: GHz')
system_arg.add_argument('--distance', type=int, default=50, help='distance between RIS and ROI along x-axis, unit: wavelength')
system_arg.add_argument('--RIS_num2', type=str, default='20,20', help='RIS element number along y- and z-axis')
system_arg.add_argument('--ROI_num3', type=str, default='1,30,30', help='voxel numbers along x- y- z-axis')
system_arg.add_argument('--Tx', type=str, default='30,50,50', help='Tx position, unit: wavelength')
system_arg.add_argument('--Rx', type=str, default='30,50,60', help='Rx position, unit: wavelength')
system_arg.add_argument('--antenna_num', type=int, default=2, help='antenna number of the TX and RX')
system_arg.add_argument('--self_interference', type=float, default=0.0, help='the ratio multiply to the TX-RX channel')


# network params
net_arg = add_argument_group('Network Params')
net_arg.add_argument('--num_glimpses', type=int, default=5, help='# certain number of glimpses')
net_arg.add_argument('--rnn_hidden_size', type=int, default=256, help='hidden size of rnn/lstm')
net_arg.add_argument('--measure_embedding_hidden_size', type=str, default='256,256', help='NN hidden size')
net_arg.add_argument('--RIS_phase_power_embedding_hidden_size', type=str, default='256,256', help='')
net_arg.add_argument('--RIS_phase_customization_hidden_size', type=str, default='256', help='')
net_arg.add_argument('--classify_hidden_size', type=str, default='256', help='')


# data params # these configs are not renewed by 'renew_test_config_from_csv_further'
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.1, help='proportion of training set used for validation')
data_arg.add_argument('--batch_size', type=int, default=128, help='')
data_arg.add_argument('--num_workers', type=int, default=8, help='')
data_arg.add_argument('--shuffle', type=str2bool, default=True, help='')
data_arg.add_argument('--train_data_scale', type=float, default=0.5, help='used percentage of training data for training')
data_arg.add_argument('--data_name', type=str, default='mnist', help='use which dataset')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=200, help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=1e-3, help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=5, help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=20, help='Number of epochs to wait before stopping train')
train_arg.add_argument('--learned_start', type=str2bool, default=True, help='If the first pattern should be learned')
train_arg.add_argument('--wandb_project', type=str, default='', help='wandb project name')
train_arg.add_argument('--ckpt_save_interval', type=int, default=5, help='save ckpt every which epochs')


# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--test_wandb_data', type=str, default='./wandb_csv/wandb_export_2025-03-01T12_00_00.000+08_00.csv', help='Load the wandb csv for prediction')
misc_arg.add_argument('--test_index', type=int, default=7, help='test which data in the wandb csv')
misc_arg.add_argument('--random_seed', type=int, default=2, help='Seed to ensure reproducibility')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/', help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False, help='Whether to resume training from checkpoint')
misc_arg.add_argument('--resume_load_model_dir', type=str, default='', help='load model dir for continuing training')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
