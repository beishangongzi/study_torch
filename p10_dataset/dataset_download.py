# create by andy at 2022/4/21
# reference: 

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

dataset_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root="./dataset",
                                         train=True,
                                         download=True,
                                         transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="./dataset",
                                        train=True,
                                        download=False,
                                        transform=dataset_transform)


test_loader = DataLoader(dataset=test_set,
                         batch_size=64,
                         shuffle=True,
                         num_workers=0,
                         drop_last=False)

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
    writer.add_images("test_batch", imgs, step)
    step += 1




# writer = SummaryWriter("logs")
# for i in range(10):
#     img, target = test_set[i]
#     writer.add_image("test_set", img, i)
#
# writer.close()



if __name__ == '__main__':
    pass
