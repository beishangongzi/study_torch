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
        x = self.model(x)
        return x


model = MyModel()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(200):
    loss_all = 0.
    step = 0
    for data in test_loader:
        if step == 10:
            break
        imgs, targets = data
        outputs = model(imgs)
        result_loss = loss(outputs, targets)
        loss_all += result_loss
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        step += 1
    print(loss_all)
if __name__ == '__main__':
    pass

if __name__ == '__main__':
    pass
