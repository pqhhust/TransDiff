import numpy as np
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.datasets
import datasets.svhn_loader as svhn
import torch
# import datasets.cifar_loader

def TrainDataLoader(img_dir, transform_train, batch_size):
    train_set = ImageFolder(img_dir, transform_train)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    return train_loader

# test data loader
def TestDataLoader(img_dir, transform_test, batch_size):
    test_set = ImageFolder(img_dir, transform_test)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    return test_loader

def get_ood_loader(dataset, test_dir, batch_size):
    if dataset == 'svhn':
        test_set = svhn.SVHN(test_dir, split='test', transform=torchvision.transforms.ToTensor(), download=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    elif dataset == 'cifar10' or dataset == 'cifar100':
        if dataset == 'cifar10':
            norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            nb_cls = 10
        elif dataset == 'cifar100':
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            nb_cls = 100

        # transformation of the test set
        transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(norm_mean, norm_std)])

        test_loader = TestDataLoader(test_dir, transform_test, batch_size)
    else:
        test_set = torchvision.datasets.ImageFolder(test_dir, transform=torchvision.transforms.Compose([torchvision.transforms.Resize(32), 
                                                                                            torchvision.transforms.CenterCrop(32), 
                                                                                            torchvision.transforms.ToTensor()]))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader

def get_loader(dataset, train_dir, val_dir, test_dir, batch_size):

    if dataset == 'cifar10':
        norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        nb_cls = 10
    elif dataset == 'cifar100':
        norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        nb_cls = 100

    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(norm_mean, norm_std)])

    # transformation of the test set
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(norm_mean, norm_std)])

    train_loader = TrainDataLoader(train_dir, transform_train, batch_size)
    val_loader = TestDataLoader(val_dir, transform_test, batch_size)
    test_loader = TestDataLoader(test_dir, transform_test, batch_size)

    return train_loader, val_loader, test_loader, nb_cls