# create by andy at 2022/4/21
# reference: 
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("logs")
# writer.add_image()
# for i in range(100):
#     writer.add_scalar("y=x", 2 * i, i)

import numpy as np
from PIL import Image
img = Image.open("../data/val/ants/desert_ant.jpg")
np_img = np.array(img)
writer.add_image("img", np_img, 2, dataformats="HWC")


writer.close()


if __name__ == '__main__':
    pass
