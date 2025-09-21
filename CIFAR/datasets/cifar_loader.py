# import numpy as np
# import torchvision.transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# # import datasets.cifar_loader
# from transformers import ViTImageProcessor

# def TrainDataLoader(img_dir, transform_train, batch_size):
#     train_set = ImageFolder(img_dir, transform_train)
#     train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
#     return train_loader

# # test data loader
# def TestDataLoader(img_dir, transform_test, batch_size):
#     test_set = ImageFolder(img_dir, transform_test)
#     test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
#     return test_loader

# def get_loader(dataset, train_dir, val_dir, test_dir, batch_size):

#     if dataset == 'cifar10':
#         norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
#         nb_cls = 10
#     elif dataset == 'cifar100':
#         norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
#         nb_cls = 100

#     # transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
#     #                                                     torchvision.transforms.RandomHorizontalFlip(),
#     #                                                     torchvision.transforms.ToTensor(),
#     #                                                     torchvision.transforms.Normalize(norm_mean, norm_std)])

#     # # transformation of the test set
#     # transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#     #                                                     torchvision.transforms.Normalize(norm_mean, norm_std)])
#     processor = ViTImageProcessor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
#     transform_test = torchvision.transforms.Compose([
#         # torchvision.transforms.RandomResizedCrop(32),
#         # torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.Lambda(lambda x: processor(x, return_tensors="pt")['pixel_values'].squeeze()),
#         # torchvision.transforms.RandomResizedCrop(224),
#         # torchvision.transforms.RandomHorizontalFlip(),
#         # torchvision.transforms.RandAugment(num_ops=2, magnitude=9),
#         # torchvision.transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
#     ])
#     transform_train = torchvision.transforms.Compose([
#         # torchvision.transforms.RandomResizedCrop(32),
#         # torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.Lambda(lambda x: processor(x, return_tensors="pt")['pixel_values'].squeeze()),
#         torchvision.transforms.RandomResizedCrop(224),
#         torchvision.transforms.RandomHorizontalFlip(),
#     ])


#     train_loader = TrainDataLoader(train_dir, transform_train, batch_size)
#     val_loader = TestDataLoader(val_dir, transform_test, batch_size)
#     test_loader = TestDataLoader(test_dir, transform_test, batch_size)

#     return train_loader, val_loader, test_loader, nb_cls

from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor


def TrainDataLoader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


def TestDataLoader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)


def get_loader(dataset, train_dir, val_dir, test_dir, batch_size):

    if dataset == 'cifar10':
        norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        nb_cls = 10
    elif dataset == 'cifar100':
        norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        nb_cls = 100

    processor = ViTImageProcessor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: processor(x, return_tensors="pt")['pixel_values'].squeeze()),
    ])
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: processor(x, return_tensors="pt")['pixel_values'].squeeze()),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
    ])

    # load datasets
    train_set = ImageFolder(train_dir, transform_train)
    val_set = ImageFolder(val_dir, transform_train)   # use train transform if merging into train
    test_set = ImageFolder(test_dir, transform_test)

    # merge train + val
    merged_train_set = ConcatDataset([train_set, val_set])

    # dataloaders
    train_loader = TrainDataLoader(merged_train_set, batch_size)
    test_loader = TestDataLoader(test_set, batch_size)

    return train_loader, test_loader, test_loader, nb_cls
