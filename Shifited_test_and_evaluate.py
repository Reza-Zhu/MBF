# -*- coding: utf-8 -*-
import glob
import os
import time
import model_
import torch
import scipy.io
import shutil
import argparse
import numpy as np
import pandas as pd
from torch import nn
from utils import fliplr, load_network, which_view, get_id, get_yaml_value
from Create_MultiModal_Dataset import Multimodel_Dateset_flip
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
def eval_and_test(query_name, config_file, net_path, save_path, gap):

    param = get_yaml_value(config_file)
    data_path = param["dataset_path"]
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    table_path = os.path.join(save_path, param["model"] + "_" + str(1652) + "_" + "shifted_query_" +
                              ".csv")

    evaluate_csv = pd.DataFrame(index=["recall@1", "recall@5", "recall@10", "recall@1p", "AP", "time"])

    image_datasets = {x: Multimodel_Dateset_flip(os.path.join(data_path, 'test', x), data_transforms, gap) for x in
                      ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=param["batch_size"],
                                                  # batch_size=16,
                                                  shuffle=False) for x in
                   ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}

    model = Hybird_ViT(701, 0.1)
    model.load_state_dict(torch.load(net_path))
    for i in range(2):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()

    model = model.eval()
    model = model.cuda()

    if "drone" in query_name:
        gallery_name = "gallery_satellite"
        query_name = "query_drone"
    else:
        gallery_name = "gallery_drone"
        query_name = "query_satellite"

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
    gallery_feature = gallery_feature.cuda()

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

    evaluate_result = 'Recall@1:%.4f Recall@5:%.4f Recall@10:%.4f Recall@top1:%.4f AP:%.4f' % (
        recall_1, recall_5, recall_10, recall_1p, AP)

    evaluate_csv["shifted_query" + "_" + str(gap) +
                 "_" + str(1652)] = \
        [float(recall_1), float(recall_5),
         float(recall_10), float(recall_1p),
         float(AP), float(0)]

    print(evaluate_csv)

    evaluate_csv.columns.name = ""
    evaluate_csv.index.name = "index"
    evaluate_csv = evaluate_csv.T
    evaluate_csv.to_csv(table_path)
    print(evaluate_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--query', type=str, default="drone", help='query set: drone or satellite')
    parser.add_argument('--cfg', type=str, default='settings.yaml', help='config file XXX.yaml path')
    parser.add_argument('--multi', type=int, default=1, help='multi number for example: if multi == 1 fusion image '
                                                             'number = 50/1 = 50')
    parser.add_argument('--weight', type=str, default="/home/sues/save_model_weight/Release_1652_2022-11-19-01:26:05/net_137.pth", help='evaluate which weight, path')
    parser.add_argument('--csv_save_path', type=str, default="./result", help="evaluation result table store path")
    parser.add_argument('--gap', type=int, default=10, help='shifted gap')
    opt = parser.parse_known_args()[0]

    eval_and_test(opt.query, opt.cfg, opt.weight, opt.csv_save_path, opt.gap)
