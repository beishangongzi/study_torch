# create by andy at 2022/4/21
# reference:
import pathlib

from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir: os.path, label_dir: os.path):
        super(MyData, self).__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.imgs = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.imgs)


class MyData2(Dataset):
    def __init__(self, data_dir: os.path, label_dir: os.path):
        super(MyData2, self).__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.imgs = os.listdir(data_dir)

    def __getitem__(self, item):
        img_name = self.imgs[item]
        img_real_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_real_path)
        label_path = os.path.join(self.label_dir, img_name.split(".")[0])
        with open(label_path, "r", encoding="urf-8") as f:
            label = f.readline().strip()
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    my_data1 = MyData("../hymenoptera_data/train", "ants")
    assert len(my_data1) == 124
    my_data2 = MyData("../hymenoptera_data/train", "bees")
    assert len(my_data2) == 121
    assert len(my_data1 + my_data2) == 245

    my_data3 = MyData2("../data/train/ants_image", "../data/train/ants_label")
    assert len(my_data3) == 124
