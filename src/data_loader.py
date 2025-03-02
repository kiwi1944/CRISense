import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import collections
import logging


def get_train_valid_loader(
    batch_size,
    valid_size=0.1,
    train_data_scale=1.0,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    ROI_num3='5,30,30',
    data_name='mnist'
):
    assert ((valid_size >= 0) and (valid_size <= 1)), '[!] valid_size should be in the range [0, 1].'
    dataset_folder = './data/'
    
    if data_name == 'mnist':
        classes = 10
        ROI_num3 = list(map(lambda x: int(x), ROI_num3.split(',')))
        ROI_num = ROI_num3[0] * ROI_num3[1] * ROI_num3[2]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5])])
        train_data_mnist = datasets.MNIST(root = dataset_folder, transform=transform, train = True, download = True)

        num_train_data = int(len(train_data_mnist) * train_data_scale) if torch.cuda.is_available() else 1000
        train_data = [0] * num_train_data
        label_count = collections.Counter()
        for i in range(num_train_data):
            img, label = train_data_mnist[i]
            img = img * 0.5 + 0.5
            x = torch.zeros(ROI_num3)
            x[ROI_num3[0] // 2, 1 : -1, 1 : -1] = img
            x = torch.reshape(x, (1, ROI_num))
            x = torch.complex(x, torch.tensor(0, dtype=torch.float32))
            train_data[i] = (x, label)
            label_count.update([label])
        logging.info(f'number of instances for each label: {sorted(label_count.items())}')

        n_val = int(len(train_data) * valid_size)
        n_train = len(train_data) - n_val
        train_set, val_set = random_split(train_data, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    else:
        raise RuntimeError('undefined dataset name: {}'.format(data_name))

    return train_loader, valid_loader, classes, len(train_set), len(val_set)

def get_test_loader(
    batch_size,
    num_workers=4,
    pin_memory=True,
    ROI_num3='5,30,30',
    data_name='mnist'
):
    dataset_folder = './data/'
    
    if data_name == 'mnist':
        classes = 10
        ROI_num3 = list(map(lambda x: int(x), ROI_num3.split(',')))
        ROI_num = ROI_num3[0] * ROI_num3[1] * ROI_num3[2]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5])])
        test_data_mnist = datasets.MNIST(root=dataset_folder, transform = transform, train = False)
        num_test_data = len(test_data_mnist) if torch.cuda.is_available() else 500
        test_data = [0] * num_test_data
        label_count = collections.Counter()
        for i in range(num_test_data):
            img, label = test_data_mnist[i]
            img = img * 0.5 + 0.5
            x = torch.zeros(ROI_num3)
            x[ROI_num3[0] // 2, 1 : -1, 1 : -1] = img
            x = torch.reshape(x, (1, ROI_num))
            x = torch.complex(x, torch.tensor(0, dtype=torch.float32))
            test_data[i] = (x, label)
            label_count.update([label])
        logging.info(f'number of instances for each label: {sorted(label_count.items())}')
        data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    else:
        raise RuntimeError('undefined dataset name: {}'.format(data_name))
    
    return data_loader, classes, num_test_data
