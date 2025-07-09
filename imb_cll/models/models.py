import torchvision
import torch.nn as nn

def get_resnet18(num_classes, input_dataset):
    # resnet = torchvision.models.resnet18(weights=None)
    resnet = torchvision.models.resnet18()
    # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # CIFAR is converted from RGB to Grayscale
    if input_dataset in ('MNIST', 'FashionMNIST', 'KMNIST'):
        num_channel = 1
    else:
        num_channel = 3
    print("------------------------------------")
    print("num channel {}, data type {}".format(num_channel, input_dataset))
    print("------------------------------------")
    resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet

def get_modified_resnet18(num_classes, input_dataset):
    resnet = torchvision.models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet