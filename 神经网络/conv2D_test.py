# create by andy at 2022/4/21
# reference:
import torchvision
import torch
from torch import nn
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
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=3,
                               stride=1,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = MyModel()

writer = SummaryWriter("./logs")

step = 0
for data in test_loader:
    imgs, targets = data
    output = model(imgs)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("input_conv_test", imgs, step)
    writer.add_images("output_conv_test", output, step)
    step += 1
    if step == 10:
        break
if __name__ == '__main__':
    pass
