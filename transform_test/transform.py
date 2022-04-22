# create by andy at 2022/4/21
# reference:
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# transform --> tensor
from torchvision.utils import _log_api_usage_once

img = Image.open("../data/val/ants/desert_ant.jpg")
img = transforms.ToTensor()(img)

writer = SummaryWriter("logs")

writer.add_image("tensor_img", img)
writer.close()


class MyTransform:

    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, pic):
        # return F.to_tensor(pic)
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


if __name__ == '__main__':
    pass
