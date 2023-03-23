# -*- coding: utf-8 -*-
import glob
import os
import time
import model_
import torch
import scipy.io
import shutil
import numpy as np
import pandas as pd
from torch import nn
from utils import fliplr, load_network, which_view, get_id, get_yaml_value
from Create_MultiModal_Dataset import Multimodel_Dateset_flip

from torchvision import datasets, models, transforms

from Multi_HBP import Hybird_ViT
if torch.cuda.is_available():
    device = torch.device("cuda:0")

def evaluate(qf, ql, gf, gl):

    query = qf.view(-1, 1)

    score = torch.mm(gf, query)

    score = score.squeeze(1).cpu()

    score = score.numpy()

    # predict index
    index = np.argsort(score)  # from small to large

    index = index[::-1]

    query_index = np.argwhere(gl == ql)

    good_index = query_index

    junk_index = np.argwhere(gl == -1)
    # print(junk_index)  = []

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):


    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    # print(cmc.shape) torch.Size([51355])
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]


    # find good_index index
    ngood = len(good_index)

    mask = np.in1d(index, good_index)


    rows_good = np.argwhere(mask == True)

    rows_good = rows_good.flatten()


    cmc[rows_good[0]:] = 1
    # print(cmc)
    # print(cmc.shape) torch.Size([51355])

    # print(cmc)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        # d_racall = 1/54
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)

        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


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

            ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(4)
        # print("fnorm", fnorm.shape)
        ff = ff.div(fnorm.expand_as(ff))
        # print("ff", ff.shape)
        ff = ff.view(ff.size(0), -1)

        features = torch.cat((features, ff.data.cpu()), 0)  # 在维度0上拼接
    return features


############################### main function #######################################
def eval_and_test(query_name, gallery_name, net_path, gap):
    data_path = get_yaml_value("dataset_path")
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {x: Multimodel_Dateset_flip(os.path.join(data_path, 'test', x), data_transforms, gap) for x in
                      ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  # batch_size=get_yaml_value("batch_size"),
                                                  batch_size=16,
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
    which_query = which_view(query_name)
    which_gallery = which_view(gallery_name)
    # image_datasets, data_loader = Multimodel_Dateset(test_data_path=data_path)
    gallery_path = image_datasets[gallery_name].imgs
    query_path = image_datasets[query_name].imgs

    gallery_label, gallery_path = get_id(gallery_path)
    query_label, query_path = get_id(query_path)

    with torch.no_grad():

        query_feature = extract_feature(model, data_loader[query_name], which_query)

        gallery_feature = extract_feature(model, data_loader[gallery_name], which_gallery)

    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label,
              'gallery_path': gallery_path, 'query_f': query_feature.numpy(),
              'query_label': query_label, 'query_path': query_path}

    scipy.io.savemat('shifted_pytorch_result.mat', result)
    print("multi_pytorch_result.mat has saved")

    result = scipy.io.loadmat("shifted_pytorch_result.mat")

    # initialize query feature data
    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]

    # initialize all(gallery) feature data
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]

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
    try:
        path = query_name + "_" + str(gap) + ".txt"
        with open(path, 'w') as f:
            f.write(evaluate_result)
            f.close()
    except:
        pass
    # show result and save
    # save_path = os.path.join(get_yaml_value("weight_save_path"), get_yaml_value('name'))

    # shutil.copy('settings.yaml', os.path.join(save_path, "settings_saved.yaml"))
    # print(round(len(gallery_label)*0.01))
    print(evaluate_result)


if __name__ == '__main__':
    gaps = [10, 20, 30, 40]

    query_name = "query_drone"
    gallery_name = "gallery_satellite"

    net_path = "/home/sues/save_model_weight/Release_final_weight/net_077.pth"
    for gap in gaps:
        eval_and_test(query_name, gallery_name, net_path, gap)

    #######
    query_name = "query_satellite"
    gallery_name = "gallery_drone"

    net_path = "/home/sues/save_model_weight/Release_final_weight/net_137.pth"
    for gap in gaps:
        eval_and_test(query_name, gallery_name, net_path, gap)
