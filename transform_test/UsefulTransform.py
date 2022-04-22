# create by andy at 2022/4/21
# reference: 

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("logs")


img = Image.open("./0013035.jpg")
print(img.size)
trans_to_to_tensor = transforms.ToTensor()
img_tensor = trans_to_to_tensor(img)
writer.add_image("norm", img_tensor, 0)

trans_norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
img_tensor = trans_norm(img_tensor)
writer.add_image("norm", img_tensor, 1)

trans_resize = transforms.Resize((512, 512))
img_tensor = trans_resize(img_tensor)
writer.add_image("norm", img_tensor, 2)
print(img_tensor.size())


trans_random = transforms.RandomCrop(100)
img_tensor = trans_random(img_tensor)
writer.add_image("norm", img_tensor, 3)



writer.close()




if __name__ == '__main__':
    pass
