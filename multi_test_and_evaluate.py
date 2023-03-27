# -*- coding: utf-8 -*-
import glob
import os
import time
import model_
import torch
import scipy.io
import argparse
import shutil
import numpy as np
import pandas as pd
from torch import nn
from utils import fliplr, load_network, which_view, get_id, get_yaml_value
from Create_MultiModal_Dataset import Multimodel_Dateset
from U1652_test_and_evaluate import evaluate

from torchvision import datasets, models, transforms

from Multi_HBP import Hybird_ViT
if torch.cuda.is_available():
    device = torch.device("cuda:0")


def extract_feature(model, dataloaders, view_index=1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, text, label = data
        n, c, h, w = img.size()
        count += n
        text = text.to(device)
        ff = torch.FloatTensor(n, 512, 4).zero_().cuda()

        # why for in range(2)：
        # 1. for flip img
        # 2. for normal img

        for i in range(2):
            if i == 1:
                img = fliplr(img)

            input_img = img.to(device)
            outputs = None
            if view_index == 1:
                outputs, _ = model(input_img, None, text, None)
            elif view_index == 2:
                _, outputs = model(None, input_img, None, text)
            # print(outputs.shape)
            # print(ff.shape)
            ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(4)
        # print("fnorm", fnorm.shape)
        ff = ff.div(fnorm.expand_as(ff))
        # print("ff", ff.shape)
        ff = ff.view(ff.size(0), -1)

        features = torch.cat((features, ff.data.cpu()), 0)  # 在维度0上拼接
    return features


############################### main function #######################################
def eval_and_test(multi_coff, config_file, weight_path, save_path):
    param = get_yaml_value(config_file)
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    table_path = os.path.join(save_path, param["model"] + "_" + str(1652) + "_" + "multi_query_" +
                              ".csv")
    evaluate_csv = pd.DataFrame(index=["recall@1", "recall@5", "recall@10", "recall@1p", "AP", "time"])

    image_datasets = {x: Multimodel_Dateset(os.path.join(param["dataset_path"], 'test', x), data_transforms) for x in
                      ['gallery_satellite', 'query_drone']}
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=16,
                                                  shuffle=False) for x in
                   ['gallery_satellite', 'query_drone']}

    query_name = "query_drone"
    gallery_name = "gallery_satellite"

    model = Hybird_ViT(701, 0.1)
    model.load_state_dict(torch.load(weight_path))

    for i in range(2):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()

    model = model.eval()
    model = model.cuda()
    which_query = which_view(query_name)
    which_gallery = which_view(gallery_name)

    gallery_path = image_datasets[gallery_name].imgs
    query_path = image_datasets[query_name].imgs

    gallery_label, gallery_path = get_id(gallery_path)
    query_label, query_path = get_id(query_path)

    with torch.no_grad():
        query_feature = extract_feature(model, data_loader[query_name], which_query)
        gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)

    # fed tensor to GPU
    query_feature = query_feature.cuda()
    new_query_feature = torch.FloatTensor().cuda()
    gallery_feature = gallery_feature.cuda()
    multi = True

    # coffs = [1, 2, 6, 18, 54]
    # University-1652
    image_per_class = 54 // multi_coff
    # coff = 54 // image_per_class

    print(image_per_class)
    query_length = len(query_label) + image_per_class

    feature_list = list(range(0, query_length, image_per_class))
    query_concat = np.ones(((len(feature_list)-1)//multi_coff, multi_coff))

    if multi:
        index = list(query_label).index
        query_label = sorted(list(set(list(query_label))), key=index)

        for i in range(len(query_label)):
            query_concat[i] = query_label[i] * query_concat[i]

        query_label = query_concat.reshape(-1,)
        # print(query_feature.shape)
        for i in range(len(feature_list)):
            if feature_list[i] == (query_length - image_per_class):
                continue

            multi_feature = torch.mean(query_feature[feature_list[i]:feature_list[i+1], :], 0)
            # print(multi_feature.shape)
            multi_feature = multi_feature.view(1, 2048)
            new_query_feature = torch.cat((new_query_feature, multi_feature), 0)

        query_feature = new_query_feature

    # CMC = recall
    CMC = torch.IntTensor(len(gallery_label)).zero_()

    # ap = average precision
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)
    # print(len(query_label))
    recall_1 = CMC[0] * 100
    recall_5 = CMC[4] * 100
    recall_10 = CMC[9] * 100
    recall_1p = CMC[round(len(gallery_label) * 0.01)] * 100
    AP = ap / len(query_label) * 100

    evaluate_csv["multi_query" + "_" + str(image_per_class) +
                 "_" + str(1652)] = \
        [float(recall_1), float(recall_5),
         float(recall_10), float(recall_1p),
         float(AP), float(0)]

    print(evaluate_csv)

    evaluate_csv.columns.name = ""
    evaluate_csv.index.name = "index"
    evaluate_csv = evaluate_csv.T
    evaluate_csv.to_csv(table_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='settings.yaml', help='config file XXX.yaml path')
    parser.add_argument('--multi', type=int, default=1, help='multi number for example: if multi == 1 fusion image '
                                                             'number = 50/1 = 50')
    parser.add_argument('--weight', type=str, default=None, help='evaluate which weight, path')
    parser.add_argument('--csv_save_path', type=str, default="./result", help="evaluation result table store path")
    opt = parser.parse_known_args()[0]

    eval_and_test(opt.multi, opt.cfg, opt.weight, opt.csv_save_path)
