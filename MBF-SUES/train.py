from __future__ import print_function, division

import time
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from pytorch_metric_learning import losses, miners


from Multi_HBP import Hybird_ViT
from utils import get_yaml_value, parameter, create_dir, save_feature_network, setup_seed
from Preprocessing import Create_Training_Datasets
import random
import os

if torch.cuda.is_available():
    device = torch.device("cuda:1")
cudnn.benchmark = True


# torch.cuda.manual_seed(random.randint(1, 100))
# setup_seed()

def one_LPN_output(outputs, labels, criterion, block):
    # part = {}
    # print(len(outputs))
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    # print(len(outputs))
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)
    _, preds = torch.max(score.data, 1)

    return preds, loss


def train(config_path):
    param_dict = get_yaml_value(config_path)
    print(param_dict)
    classes = param_dict["classes"]
    num_epochs = param_dict["num_epochs"]
    drop_rate = param_dict["drop_rate"]
    lr = param_dict["lr"]
    weight_decay = param_dict["weight_decay"]
    model_name = param_dict["model"]
    fp16 = param_dict["fp16"]
    weight_save_path = param_dict["weight_save_path"]
    LPN = param_dict["LPN"]
    batchsize = param_dict["batch_size"]
    height = param_dict["height"]
    data_path = param_dict["dataset_path"]
    block = param_dict["block"]
    image_size = param_dict["image_size"]


    all_block = block
    train_data_path = data_path + "/Training/{}".format(height)

    dataloaders, image_datasets = Create_Training_Datasets(train_data_path=train_data_path, batch_size=batchsize, image_size=image_size)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}

    model = Hybird_ViT(classes, drop_rate, all_block).to(device)

    if LPN:
        ignored_params = list()
        for i in range(all_block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1 * lr}]
        for i in range(all_block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': lr})
        optimizer = optim.SGD(optim_params, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        # opt = torchcontrib.optim.SWA(optimizer)
    else:
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=weight_decay, momentum=0.9, nesterov=True)

    if fp16:
        # from apex.fp16_utils import *
        from apex import amp, optimizers
        model, optimizer_ft = amp.initialize(model, optimizer, opt_level="O2")

    criterion = nn.CrossEntropyLoss()
    # criterion1 = nn.KLDivLoss()
    # circle = circle_loss.CircleLoss(m=0.4, gamma=80)
    criterion_func = losses.TripletMarginLoss(margin=0.3)
    miner = miners.MultiSimilarityMiner()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    print("Dataloader Preprocessing Finished...")
    MAX_LOSS = 10
    print("Training Start >>>>>>>>")
    weight_save_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    dir_model_name = model_name + "_" + str(height) + "_" + weight_save_name
    save_path = os.path.join(weight_save_path, dir_model_name)
    create_dir(save_path)
    print(save_path)
    parameter("name", dir_model_name)

    warm_epoch = 5
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / batchsize) * warm_epoch  # first 5 epoch

    for epoch in range(num_epochs):
        since = time.time()

        running_loss = 0.0
        running_corrects1 = 0.0
        running_corrects2 = 0.0
        total1 = 0.0
        total2 = 0.0
        model.train(True)
        for data1, data2 in zip(dataloaders["satellite"], dataloaders["drone"]):

            input1, text1, label1 = data1
            input2, text2, label2 = data2

            input1, input2 = input1.to(device), input2.to(device)
            text1, text2 = text1.to(device), text2.to(device)
            label1, label2 = label1.to(device), label2.to(device)

            total1 += label1.size(0)
            total2 += label2.size(0)

            optimizer.zero_grad()

            output1, output2, feature1, feature2 = model(input1, input2, text1, text2)

            fnorm = torch.norm(feature1, p=2, dim=1, keepdim=True) * np.sqrt(all_block + 2)
            fnorm2 = torch.norm(feature2, p=2, dim=1, keepdim=True) * np.sqrt(all_block + 2)
            # fnorm3 = torch.norm(feature3, p=2, dim=1, keepdim=True) * np.sqrt(all_block)
            # fnorm4 = torch.norm(feature4, p=2, dim=1, keepdim=True) * np.sqrt(all_block)

            feature1 = feature1.div(fnorm.expand_as(feature1))
            feature2 = feature2.div(fnorm2.expand_as(feature2))
            loss1 = loss2 = loss3 = loss4 = loss6 = loss5 = loss7 = loss8 = 0

            if LPN:
                # print(len(output1))
                preds1, loss1 = one_LPN_output(output1[2:], label1, criterion, all_block)
                preds2, loss2 = one_LPN_output(output2[2:], label2, criterion, all_block)

                loss3 = criterion(output1[1], label1)
                loss4 = criterion(output2[1], label2)

                loss7 = criterion(output1[0], label1)
                loss8 = criterion(output2[0], label2)
                # _, preds1 = torch.max(output1[1].data, 1)
                # _, preds2 = torch.max(output2[1].data, 1)
                # print(loss)
            else:
                loss1 = criterion(output1[0], label1)
                loss2 = criterion(output2[1], label2)
                loss3 = criterion(output1[0], label1)
                loss4 = criterion(output2[1], label2)

                _, preds1 = torch.max(output1[0].data, 1)
                _, preds2 = torch.max(output2[1].data, 1)
                _, preds3 = torch.max(output1[0].data, 1)
                _, preds4 = torch.max(output2[1].data, 1)

            # Identity loss
            loss = loss1 + loss2 + loss3 + loss4 + loss7 + loss8

            # Triplet loss
            hard_pairs = miner(feature1, label1)
            hard_pairs2 = miner(feature2, label2)
            loss += criterion_func(feature1, label1, hard_pairs) + \
                    criterion_func(feature2, label2, hard_pairs2)


            if epoch < warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up
            if fp16:  # we use optimizer to backward loss
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # pass
            else:
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects1 += preds1.eq(label1.data).sum()
            running_corrects2 += preds2.eq(label2.data).sum()
            # print(loss.item(), preds1.eq(label1.data).sum(), preds2.eq(label2.data).sum())

        scheduler.step()
        epoch_loss = running_loss / classes
        satellite_acc = running_corrects1 / total1
        drone_acc = running_corrects2 / total2
        time_elapsed = time.time() - since

        print('[Epoch {}/{}] {} | Loss: {:.4f} | Drone_Acc: {:.2f}% | Satellite_Acc: {:.2f}% | Time: {:.2f}s' \
              .format(epoch + 1, num_epochs, "Train", epoch_loss, drone_acc * 100, satellite_acc * 100, time_elapsed))

        if drone_acc > 0.95 and satellite_acc > 0.95:
            if epoch_loss < MAX_LOSS and epoch > (num_epochs - 15):
                MAX_LOSS = epoch_loss
                save_feature_network(model, dir_model_name, epoch + 1)
                print(model_name + " Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='settings.yaml', help='config file XXX.yaml path')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt(True)
    print(opt.cfg)
    train(opt.cfg)
