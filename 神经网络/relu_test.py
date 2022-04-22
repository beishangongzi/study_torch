# create by andy at 2022/4/21
# reference: 
import torch

input_tensor = torch.tensor([[1, -0.5],
                             [-1, 3]])

input_tensor = torch.reshape(input_tensor, (1, 1, 2, 2))

class MyModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        return self.relu(input)

if __name__ == '__main__':
    model = MyModel()
    res = model(input_tensor)
    print(res)
