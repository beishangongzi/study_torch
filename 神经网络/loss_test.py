# create by andy at 2022/4/21
# reference: 
import torch

x = torch.tensor([1, 2, 3], dtype=torch.float)
y = torch.tensor([1, 2, 6], dtype=torch.float)
x = torch.reshape(x, (1, 1, 1, 3))
y = torch.reshape(y, (1, 1, 1, 3))

loss = torch.nn.L1Loss(reduction="sum")
loss_value = loss(x, y)
print(loss_value)


loss = torch.nn.L1Loss()
loss_value = loss(x, y)
print(loss_value)


loss = torch.nn.MSELoss()
loss_value = loss(x, y)
print(loss_value)



x = torch.tensor([0.1, 0.8, 0.1])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss = torch.nn.CrossEntropyLoss()
loss_value = loss(x, y)
print(loss_value)




if __name__ == '__main__':
    pass
