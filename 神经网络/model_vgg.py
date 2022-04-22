# create by andy at 2022/4/21
# reference:
import os

import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset_transform = transforms.Compose([
    transforms.ToTensor(),
])
assert os.path.exists("../p10_dataset/dataset/")
test_set = torchvision.datasets.CIFAR10(root="../p10_dataset/dataset/",
                                         train=False,
                                         download=False,
                                         transform=dataset_transform)

train_set = torchvision.datasets.CIFAR10(root="../p10_dataset/dataset/",
                                          train=True,
                                          download=False,
                                          transform=dataset_transform)

test_loader = DataLoader(dataset=test_set,
                         batch_size=64,
                         shuffle=True,
                         num_workers=0,
                         drop_last=False)

vgg16 = torchvision.models.vgg16(pretrained=True)
# vgg16.add_module("add_linear", nn.Linear(1000, 10))
vgg16.classifier[6] = nn.Linear(4096, 10)
print(vgg16)


if __name__ == '__main__':
    pass
