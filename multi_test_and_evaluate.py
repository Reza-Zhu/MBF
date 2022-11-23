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
from Create_MultiModal_Dataset import Multimodel_Dateset

from torchvision import datasets, models, transforms

from Multi_HBP import Hybird_ViT
if torch.cuda.is_available():
    device = torch.device("cuda:0")

def evaluate(qf, ql, gf, gl):
    # print(qf.shape) torch.Size([512])
    # print(gf.shape) torch.Size([51355, 512])
    # print(ql) 0 ()
    # print(gl) [0,0...0] len = 51355 shape = (51355,)

    query = qf.view(-1, 1)
    # print(query.shape)  query.shape = (512,1)
    # gf.shape = (51355, 512)
    # 矩阵相乘

    # score 是否可理解为当前余弦距离的排序？
    score = torch.mm(gf, query)
    # score.shape = (51355,1)
    score = score.squeeze(1).cpu()
    # score.shape = （51355,)
    score = score.numpy()
    # print(score)
    # print(score.shape)

    # predict index
    index = np.argsort(score)  # from small to large
    # 从小到大的索引排列
    # print("index before", index)
    index = index[::-1]
    # print("index after", index)
    # 从大到小的索引排列

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # print(query_index.shape) (54, 1)
    # gl = ql 返回标签值相同的索引矩阵
    # 得到 ql：卫星图标签，gl：无人机图标签
    # 即 卫星图标签在 gl中的索引位置 组成的矩阵
    good_index = query_index

    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)
    # print(junk_index)  = []

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    # CMC就是recall的，只要前K里面有一个正确答案就算recall成功是1否则是0
    # mAP是传统retrieval的指标，算的是 recall和precision曲线，这个曲线和x轴的面积。

    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    # print(cmc.shape) torch.Size([51355])
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    # print(index.shape) (51355,)
    # if junk_index == []
    # return index fully

    # find good_index index
    ngood = len(good_index)
    # print("good_index", good_index) (54, 1)
    # print(index)
    # print(good_index)
    mask = np.in1d(index, good_index)
    # print(mask)
    # print(mask.shape)  (51355,)
    # 51355 中 54 个对应元素变为了True

    rows_good = np.argwhere(mask == True)
    # print(rows_good.shape) (54, 1)
    # rows_good 得到这 54 个为 True 元素的索引位置

    rows_good = rows_good.flatten()
    # print(rows_good.shape)  (54,)
    # print(rows_good[0])

    cmc[rows_good[0]:] = 1
    # print(cmc)
    # print(cmc.shape) torch.Size([51355])

    # print(cmc)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        # d_racall = 1/54
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        # n/sum
        # print("row_good[]", i, rows_good[i])
        # print(precision)
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
def eval_and_test(image_per_class):
    data_path = get_yaml_value("dataset_path")
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {x: Multimodel_Dateset(os.path.join(data_path, 'test', x), data_transforms) for x in
                      ['gallery_satellite', 'query_drone']}
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=16,
                                                  shuffle=False) for x in
                   ['gallery_satellite', 'query_drone']}
    if not os.path.exists("U1652_pytorch_result.mat"):
        multi = True

        query_name = "query_drone"
        gallery_name = "gallery_satellite"

        net_path = "/home/sues/save_model_weight/Release_final_weight/net_077.pth"
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

            scipy.io.savemat('multi_pytorch_result.mat', result)
            print("multi_pytorch_result.mat has saved")
    else:
        result = scipy.io.loadmat("U1652_pytorch_result.mat")

        # initialize query feature data
        query_feature = torch.FloatTensor(result['query_f'])
        query_label = result['query_label'][0]

        # initialize all(gallery) feature data
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_label = result['gallery_label'][0]

        # fed tensor to GPU
        query_feature = query_feature.cuda()
        new_query_feature = torch.FloatTensor().cuda()
        gallery_feature = gallery_feature.cuda()
        multi = True

        # coffs = [1, 2, 6, 18, 54]
        image_per_class = image_per_class
        coff = 54 // image_per_class

        print(image_per_class)
        query_length = len(query_label) + image_per_class

        feature_list = list(range(0, query_length, image_per_class))
        query_concat = np.ones(((len(feature_list)-1)//coff, coff))
        # print(query_concat.shape)

        if multi:
            index = list(query_label).index
            query_label = sorted(list(set(list(query_label))), key=index)

            # query_label = np.intersect1d(query_label, gallery_label)
            # print(query_label.shape)
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

        evaluate_result = 'Recall@1:%.4f Recall@5:%.4f Recall@10:%.4f Recall@top1:%.4f AP:%.4f' % (
            recall_1, recall_5, recall_10, recall_1p, AP)

        path = "multi" + "_" + str(image_per_class) + ".txt"
        with open(path, 'w') as f:
            f.write(evaluate_result)
            f.close()
        # show result and save
        # save_path = os.path.join(get_yaml_value("weight_save_path"), get_yaml_value('name'))

        # shutil.copy('settings.yaml', os.path.join(save_path, "settings_saved.yaml"))
        # print(round(len(gallery_label)*0.01))
        print(evaluate_result)



if __name__ == '__main__':
    for per_class in [54, 27, 18, 9, 3, 2]:
        eval_and_test(per_class)
