import torch
import os
from utils import get_yaml_value
from torchvision import datasets, transforms
from Create_MultiModal_Dataset import Multimodel_Dateset


def Create_Training_Datasets(train_data_path, batch_size, image_size):
    training_data_loader = {}
    transform_drone_list = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transforms_satellite_list = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    image_dataset = {}
    image_dataset["drone"] = Multimodel_Dateset(os.path.join(train_data_path, "drone"),
                                                transform=transforms.Compose(transform_drone_list))
    image_dataset["satellite"] = Multimodel_Dateset(os.path.join(train_data_path, "satellite"),
                                                    transform=transforms.Compose(transforms_satellite_list))

    training_data_loader["drone"] = torch.utils.data.DataLoader(image_dataset["drone"],
                                                                      batch_size=batch_size,
                                                                      shuffle=True,
                                                                      # num_workers=4,  # 多进程
                                                                      pin_memory=True)  # 锁页内存

    training_data_loader["satellite"] = torch.utils.data.DataLoader(image_dataset["satellite"],
                                                                          batch_size=batch_size,
                                                                          shuffle=True,
                                                                          # num_workers=4,  # 多进程
                                                                          pin_memory=True)  # 锁页内存

    return training_data_loader, image_dataset


def Create_Testing_Datasets(test_data_path, batch_size, image_size):
    print(test_data_path)
    testing_data_loader = {}
    image_datasets = {}
    transforms_test_list = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    image_datasets['query_drone'] = Multimodel_Dateset(os.path.join(test_data_path, "query_drone"),
                                                         transform=transforms.Compose(transforms_test_list))

    image_datasets['query_satellite'] = Multimodel_Dateset(os.path.join(test_data_path, "query_satellite"),
                                                             transform=transforms.Compose(transforms_test_list))

    image_datasets['gallery_drone'] = Multimodel_Dateset(os.path.join(test_data_path, "gallery_drone"),
                                                           transform=transforms.Compose(transforms_test_list))

    image_datasets['gallery_satellite'] = Multimodel_Dateset(os.path.join(test_data_path, "gallery_satellite"),
                                                               transform=transforms.Compose(transforms_test_list))

    testing_data_loader["query_drone"] = torch.utils.data.DataLoader(image_datasets['query_drone'],
                                                                     batch_size=batch_size,
                                                                     shuffle=False,
                                                                     # num_workers=4,  # 多进程
                                                                     pin_memory=True)

    testing_data_loader["query_satellite"] = torch.utils.data.DataLoader(image_datasets['query_satellite'],
                                                                         batch_size=batch_size,
                                                                         shuffle=False,
                                                                         # num_workers=4,  # 多进程
                                                                         pin_memory=True)  # 锁页内存

    testing_data_loader["gallery_drone"] = torch.utils.data.DataLoader(image_datasets['gallery_drone'],
                                                                       batch_size=batch_size,
                                                                       shuffle=False,
                                                                       # num_workers=4,  # 多进程
                                                                       pin_memory=True)  # 锁页内存

    testing_data_loader["gallery_satellite"] = torch.utils.data.DataLoader(image_datasets['gallery_satellite'],
                                                                           batch_size=batch_size,
                                                                           shuffle=False,
                                                                           # num_workers=4,  # 多进程
                                                                           pin_memory=True)  # 锁页内存

    return testing_data_loader, image_datasets


if __name__ == "__main__":
    # Cross_Dataset("../Datasets/SUES-200/Training/150", 224)
    dataloaders, img_dataset = Create_Training_Datasets(train_data_path="../SUES-200-512x512/Training/150",
                                                        batch_size=4,
                                                        image_size=384)
    # print(image_datasets['drone_train'].classes)
    # print(img_dataset['drone'].imgs)
    for img, text, label in dataloaders["drone"]:
        print(text.shape)
        print(img.shape, label.shape)
        break

