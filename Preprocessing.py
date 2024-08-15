import torch
import os
from utils import get_yaml_value, parameter, create_dir, save_feature_network
from torchvision import datasets, transforms
from Create_MultiModal_Dataset import Multimodel_Dateset


def create_U1652_dataloader(data_dir, batch_size, image_size):
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((image_size, image_size), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomPerspective(),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_satellite_list = [
        transforms.Resize((image_size, image_size), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((image_size, image_size)),
        # transforms.RandomPerspective(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'satellite': transforms.Compose(transform_satellite_list)}

    image_datasets = {}
    image_datasets['satellite'] = Multimodel_Dateset(os.path.join(data_dir, 'train', 'satellite'),
                                                       data_transforms['satellite'])
    image_datasets['drone'] = Multimodel_Dateset(os.path.join(data_dir, 'train', 'drone'),
                                                   data_transforms['train'])
    dataloaders = {}
    dataloaders['satellite'] = torch.utils.data.DataLoader(image_datasets['satellite'], batch_size=batch_size,
                                                  shuffle=True)

    dataloaders['drone'] = torch.utils.data.DataLoader(image_datasets['drone'], batch_size=batch_size,
                                                           shuffle=True)
    return dataloaders, image_datasets


if __name__ == "__main__":
    # Cross_Dataset("../Datasets/SUES-200/Training/150", 224)
    dataloaders, image_datasets = create_U1652_dataloader()
    print(image_datasets['drone'].classes)
    for img, text, label in dataloaders['drone']:
        print(text)
        print(img, label)
        break
    # U1652_path = "/media/data1/University-Release/University-Release/train"
    # Cross_Dataset_1652(U1652_path, 224)

