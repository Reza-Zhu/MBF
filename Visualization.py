import os
import re
import glob
import torch
import model_
from shutil import copyfile, copy
import random
import scipy
import pandas as pd
import numpy as np
from torch import nn
# from evaluation_methods import select_best_weight
from utils import get_yaml_value, which_view, create_dir
from U1652_test_and_evaluate import extract_feature
# from Preprocessing import
from Create_MultiModal_Dataset import Multimodel_Dateset
from Multi_HBP import Hybird_ViT
from torchvision import datasets, models, transforms


def get_rank(query_name, gallery_name):
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_path = get_yaml_value("dataset_path")


    gallery_drone_path = os.path.join(data_path, "test", "gallery_drone")
    gallery_satellite_path = os.path.join(data_path, "test", "gallery_satellite")
    gallery_drone_list = glob.glob(os.path.join(gallery_drone_path, "*"))
    gallery_drone_list = sorted(gallery_drone_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))


    gallery_satellite_list = glob.glob(os.path.join(gallery_satellite_path, "*"))
    gallery_satellite_list = sorted(gallery_satellite_list, key=lambda x: int(re.findall("[0-9]+", x[-4:])[0]))
    drone_list = []
    satellite_list = []

    if "drone" in gallery_name:
        for drone_img in gallery_drone_list:
            img_list = glob.glob(os.path.join(drone_img, "*"))
            img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x.split('/')[-1])[0]))
            for img in img_list:
                drone_list.append(img)
    elif "satellite" in gallery_name:
        for satellite_img in gallery_satellite_list:
            img_list = glob.glob(os.path.join(satellite_img, "*"))
            img_list = sorted(img_list, key=lambda x: int(re.findall("[0-9]+", x.split('/')[-1])[0]))
            for img in img_list:
                satellite_list.append(img)

    image_datasets = {x: Multimodel_Dateset(os.path.join(data_path, 'test', x), data_transforms) for x in
                      ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  # batch_size=get_yaml_value("batch_size"),
                                                  batch_size=4,
                                                  shuffle=False) for x in
                   ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}
    net_path = "/home/sues/save_model_weight/Release_final_weight/net_077.pth"

    which_query = which_view(query_name)
    which_gallery = which_view(gallery_name)
    print(net_path)
    model = Hybird_ViT(701, 0.1).cuda()
    model.load_state_dict(torch.load(net_path))
    for i in range(2):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()
    model = model.eval()

    if not os.path.exists("visual_pytorch_result.mat"):
        query_feature = extract_feature(model, data_loader[query_name], 2, 1, which_query)
        gallery_feature = extract_feature(model, data_loader[gallery_name], 2, 1, which_gallery)
        # result = scipy.io.loadmat("U1652_pytorch_result.mat")
        result = {'gallery_f': gallery_feature.numpy(), 'query_f': query_feature.numpy()}
        scipy.io.savemat('visual_pytorch_result.mat', result)
    # else:
    #     result = scipy.io.loadmat("visual_pytorch_result.mat")

    result = scipy.io.loadmat("visual_pytorch_result.mat")
    # initialize query feature data
    query_feature = torch.FloatTensor(result['query_f'])


    gallery_feature = torch.FloatTensor(result['gallery_f'])

    # gallery_feature = np.array(gallery_feature)
    # gallery_features = np.array_split(gallery_feature, len(gallery_label))
    # gallery_feature = torch.FloatTensor()
    # label_index = np.argsort(gallery_label)
    # label_index = label_index[::-1]
    # for i in label_index:
    #     gallery_feature = torch.cat([gallery_feature, torch.from_numpy(gallery_features[i])])
    # gallery_features = sorted(gallery_features, key=label_index)
    # gallery_feature = np.stack(gallery_features)
    query_img_list = image_datasets[query_name].imgs
    gallery_img_list = image_datasets[gallery_name].imgs
    matching_table = {}
    random_sample_list = random.sample(range(0, len(query_img_list)), 10)
    print(random_sample_list)
    for i in random_sample_list:
        query = query_feature[i].view(-1, 1)
        score = torch.mm(gallery_feature, query)
        score = score.squeeze(1).cpu()
        index = np.argsort(score.numpy())
        index = index[::-1].tolist()
        max_score_list = index[0:10]
        query_img = query_img_list[i][0]
        most_correlative_img = []
        for index in max_score_list:
            if "satellite" in query_name:
                most_correlative_img.append(gallery_img_list[index][0])
            elif "drone" in query_name:
                most_correlative_img.append(gallery_img_list[index][0])
        matching_table[query_img] = most_correlative_img
    matching_table = pd.DataFrame(matching_table)
    print(matching_table)
    save_path = query_name.split("_")[-1] + "_" + str(1652) + "_matching.csv"
    matching_table.to_csv(save_path)
    return save_path

def summary_csv_extract_pic(csv_path):
    csv_table = pd.read_csv(csv_path, index_col=0)
    create_dir("result")

    csv_path = os.path.join("result", csv_path.split("_")[-3])
    create_dir(csv_path)
    query_pic = list(csv_table.columns)
    for pic in query_pic:
        dir_path = os.path.join(csv_path, pic.split("/")[-4] + "_" + pic.split("/")[-3])
        create_dir(dir_path)
        dir_path = os.path.join(dir_path, pic.split("/")[-2])
        create_dir(dir_path)
        copy(pic, dir_path)
        gallery_list = list(csv_table[pic])
        print(gallery_list)
        count = 0
        for gl_path in gallery_list:
            print(gl_path)
            copy(gl_path, dir_path)
            src_name = os.path.join(dir_path, gl_path.split("/")[-1])
            dest_name = os.path.dirname(src_name) + os.sep + str(count) + "_" + gl_path.split("/")[-2] + "." + gl_path.split(".")[-1]
            print(src_name)
            print(dest_name)
            os.rename(src_name, dest_name)
            count = count + 1


if __name__ == '__main__':

    # query_name = 'query_satellite'
    # gallery_name = 'gallery_drone'
    #
    # path = get_rank(query_name, gallery_name)
    # summary_csv_extract_pic(path)

    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'

    path = get_rank(query_name, gallery_name)
    summary_csv_extract_pic(path)