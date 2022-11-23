import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from PIL import Image
import json


class Multimodel_Dateset(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms):
        self.transforms = transforms
        self.img_data_path = data_path

        if "drone" in os.path.basename(self.img_data_path):
            self.text_path = os.path.join(os.path.dirname(data_path), "text_drone")
        elif "satellite" in os.path.basename(self.img_data_path):
            self.text_path = os.path.join(os.path.dirname(data_path), "text_satellite")
            self.tensor = torch.load(os.path.join(self.text_path, "satellite.pth"))

        img_list = glob.glob(os.path.join(data_path, "*"))
        self.classes = os.listdir(data_path)
        self.img_names = []
        for imgs in img_list:
            self.img_names += glob.glob(os.path.join(imgs, '*'))
            len_img = len(glob.glob(os.path.join(imgs, '*')))
        self.labels = range(len(img_list))
        img_arr = np.array(self.labels).reshape(1, -1)
        img_arr = np.repeat(img_arr, len_img).tolist()

        self.imgs = list(zip(self.img_names, img_arr))
        # print(imgs[:10])
        # for img_dir in img_list:
        #     for img_file in glob.glob(os.path.join(img_dir, "*")):

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img = self.img_names[item]
        # text = self.text[os.path.basename(self.img_names[item])]
        if "drone" in os.path.basename(self.img_data_path):
            name = os.path.basename(img).split('.')[0] + '.pth'
            text = torch.load(os.path.join(self.text_path, name)).cpu()
        elif "satellite" in os.path.basename(self.img_data_path):
            text = self.tensor.cpu()
        # print(text.device)
        label = self.labels[self.classes.index(os.path.basename(os.path.dirname(img)))]
        # print(img, label)
        img = Image.open(img).convert('RGB')
        img = self.transforms(img)
        return img, text, label

class Multimodel_Dateset_flip(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms, gap):
        self.transforms = transforms
        self.img_data_path = data_path
        self.gap = gap
        if "drone" in os.path.basename(self.img_data_path):
            self.text_path = os.path.join(os.path.dirname(data_path), "text_drone")
        elif "satellite" in os.path.basename(self.img_data_path):
            self.text_path = os.path.join(os.path.dirname(data_path), "text_satellite")
            self.tensor = torch.load(os.path.join(self.text_path, "satellite.pth"))

        img_list = glob.glob(os.path.join(data_path, "*"))
        self.classes = os.listdir(data_path)
        self.img_names = []
        for imgs in img_list:
            self.img_names += glob.glob(os.path.join(imgs, '*'))
            len_img = len(glob.glob(os.path.join(imgs, '*')))
        self.labels = range(len(img_list))
        img_arr = np.array(self.labels).reshape(1, -1)
        img_arr = np.repeat(img_arr, len_img).tolist()

        self.imgs = list(zip(self.img_names, img_arr))
        # print(imgs[:10])
        # for img_dir in img_list:
        #     for img_file in glob.glob(os.path.join(img_dir, "*")):

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img = self.img_names[item]
        # text = self.text[os.path.basename(self.img_names[item])]
        if "drone" in os.path.basename(self.img_data_path):
            name = os.path.basename(img).split('.')[0] + '.pth'
            text = torch.load(os.path.join(self.text_path, name)).cpu()
        elif "satellite" in os.path.basename(self.img_data_path):
            text = self.tensor.cpu()
        # print(text.device)
        label = self.labels[self.classes.index(os.path.basename(os.path.dirname(img)))]
        # print(img, label)
        img = Image.open(img).convert('RGB')
        height = img.height
        flip = img.crop((0, 0, self.gap, height))
        img = img.crop((0, 0, height - self.gap, height))
        flip = flip.transpose(Image.FLIP_LEFT_RIGHT)
        joint = Image.new("RGB", (height, height))
        joint.paste(flip, (0, 0, self.gap, height))
        joint.paste(img, (self.gap, 0, height, height))
        img = self.transforms(joint)
        # plt.figure("black")
        # plt.imshow(joint)
        # plt.show()
        return img, text, label


if __name__ == "__main__":
    path = "/home/sues/media/disk2/University-Release-MultiModel/University-Release/test/gallery_satellite"
    print(os.path.basename(path))

    transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
                                    ])
    dataset = Multimodel_Dateset(path, transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    for img, text, label in loader:
        print(img.shape)
        print(text.shape)
        print(label)
        break
