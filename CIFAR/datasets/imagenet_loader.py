import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def TrainDataLoader(img_dir, transform_train, batch_size):
    train_set = ImageFolder(img_dir, transform_train)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    return train_loader

def TestDataLoader(img_dir, transform_test, batch_size):
    test_set = ImageFolder(img_dir, transform_test)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    return test_loader

def get_loader(dataset, train_dir, val_dir, test_dir, batch_size):
    if dataset == 'imagenet1k':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nb_cls = 1000
    else:
        raise ValueError("Unsupported dataset: " + dataset)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_loader = TrainDataLoader(train_dir, transform_train, batch_size)
    val_loader = TestDataLoader(val_dir, transform_test, batch_size)
    test_loader = TestDataLoader(test_dir, transform_test, batch_size)
    return train_loader, val_loader, test_loader, nb_cls

if __name__ == '__main__':
    # Update these paths to point to your preprocessed directories.
    train_dir = 'IMAGENET1K_32/train'
    val_dir = 'IMAGENET1K_32/val'
    test_dir = 'IMAGENET1K_32/test'
    batch_size = 64

    train_loader, val_loader, test_loader, nb_cls = get_loader('imagenet1k', train_dir, val_dir, test_dir, batch_size)
    for images, labels in train_loader:
        print(images.shape, labels)
        break