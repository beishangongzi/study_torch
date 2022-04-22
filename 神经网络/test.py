# create by andy at 2022/4/22
# reference:
import torch
from PIL import Image
import torchvision
from torch import nn
from torch.nn import Sequential

img_path = "imgs/img.png"
img = Image.open(img_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
])

img = transform(img)

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

model = torch.load("cifar_10_4692.pkl")
print(model)

image = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    out = model(image)

print(out.argmax(1))


if __name__ == '__main__':
    pass
