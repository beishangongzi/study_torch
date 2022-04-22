# create by andy at 2022/4/21
# reference: 

# create by andy at 2022/4/21
# reference:
import torchvision
import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset_transform = transforms.Compose([
    transforms.ToTensor(),
])
test_set = torchvision.datasets.CIFAR10(root="./../p10_dataset/dataset",
                                        train=True,
                                        download=False,
                                        transform=dataset_transform)

test_loader = DataLoader(dataset=test_set,
                         batch_size=64,
                         shuffle=True,
                         num_workers=0,
                         drop_last=False)


class MyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)

        self.model = Sequential(
        nn.Conv2d(3, 32, 5, padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 5, padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5, padding=2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1024, 64),
        nn.Linear(64, 10),
        )




    def forward(self, x):

        # x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model(x)
        return x


model = MyModel()
loss = nn.CrossEntropyLoss()
step = 0
for data in test_loader:
    if step == 10:
        break
    imgs, targets = data
    outputs = model(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)
    step += 1


if __name__ == '__main__':
    pass

if __name__ == '__main__':
    pass
