# create by andy at 2022/4/21
# reference:
import torch
from torch import nn


class MyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


if __name__ == '__main__':
    andy = MyModel()
    x = torch.tensor(1.0)
    output = andy(x)
    print(output)
