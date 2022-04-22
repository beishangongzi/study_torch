# create by andy at 2022/4/22
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
train_set = torchvision.datasets.CIFAR10(root="./../p10_dataset/dataset",
                                         train=True,
                                         download=False,
                                         transform=dataset_transform)

test_set = torchvision.datasets.CIFAR10(root="./../p10_dataset/dataset",
                                        train=False,
                                        download=False,
                                        transform=dataset_transform)
test_loader = DataLoader(dataset=test_set,
                         batch_size=64,
                         shuffle=False,
                         num_workers=0,
                         drop_last=False)

train_loader = DataLoader(dataset=train_set,
                          batch_size=64,
                          shuffle=False,
                          num_workers=0,
                          drop_last=False)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = MyModel()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    total_step = 0
    total_test_step = 0

    writer = SummaryWriter("logs")

    epoch = 10

    for i in range(epoch):
        print(f"----------{i} begin---------")
        model.train()
        for data in train_loader:
            imgs, target = data
            output = model(imgs)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_step += 1
            print(f"{loss.item()}")
            acc = (output.argmax(1) == target).sum()
            writer.add_scalar("train_loss_cifar10",
                              loss.item(),
                              total_test_step)
            total_test_step += 1
        torch.save(model, f"cifar_10_{total_test_step}.pkl")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for data in test_loader:
                img, target = data
                output = model(img)
                loss = loss_fn(output, target)
                total_loss += loss.item()
            print(total_loss)

            writer.add_scalar("test_loss_cifar10",
                              total_test_step)
    writer.close()