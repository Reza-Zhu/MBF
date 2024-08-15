import os
import timm
import time
import math
import torch
import torch.nn as nn
from torch.nn import init, functional
from torchvision import models


class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad=True).cuda()
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = x.cuda()
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'



class ClassBlock(nn.Module):

    def __init__(self, input_dim, class_num, drop_rate, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [
            nn.Linear(input_dim, num_bottleneck),
            nn.GELU(),
            nn.BatchNorm1d(num_bottleneck),
            nn.Dropout(p=drop_rate)
        ]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):

        x = self.add_block(x)
        feature = x
        x = self.classifier(x)
        return x, feature

class ResNet(nn.Module):
    def __init__(self, class_num, drop_rate, share_weight=False):
        super(ResNet, self).__init__()
        self.model_1 = timm.create_model("resnet50", pretrained=True, num_classes=0)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("resnet50", pretrained=True, num_classes=0)

        self.classifier = ClassBlock(2048, class_num, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)

        return y1, y2


class SEResNet_50(nn.Module):
    def __init__(self, classes, drop_rate, share_weight = False):
        super(SEResNet_50, self).__init__()
        self.model_1 = timm.create_model("seresnet50", pretrained=True, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("seresnet50", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(2048, classes, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)
        return y1, y2


class DenseNet(nn.Module):
    def __init__(self, class_num, drop_rate, share_weight=False):
        super(DenseNet, self).__init__()
        self.model_1 = timm.create_model("densenet201", pretrained=True, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("densenet201", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(1920, class_num, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)
        return y1, y2


class Hybird_ViT(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=True):
        super(Hybird_ViT, self).__init__()
        self.model_1 = timm.create_model("vit_base_r50_s16_384", pretrained=True, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("vit_base_r50_s16_384", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(768, classes, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
            f1 = None
        else:
            x1 = self.model_1(x1)
            y1, f1 = self.classifier(x1)

        if x2 is None:
            y2 = None
            f2 = None
        else:
            x2 = self.model_2(x2)
            y2, f2 = self.classifier(x2)
        if self.training:
            return y1, y2, f1, f2
        else:
            return f1, f2


class ViT(nn.Module):
    def __init__(self, classes, drop_rate, share_weight):
        super(ViT, self).__init__()
        # checkpoint = torch.load(os.path.join("checkpoint", "drone_checkpoint.pth"))

        self.model_1 = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        # self.model_1.load_state_dict(checkpoint)
        # model.model_2.load_state_dict(checkpoint["model"])

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

        # self.model_2 = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        # self.model_1.load_state_dict(torch.load("../SS-Study/checkpoint/satellite_checkpoint.pth"))
        # self.model_2.load_state_dict(torch.load("../SS-Study/checkpoint/drone_checkpoint.pth"))
        # self.model_1 = self.model_2

        self.bn = torch.nn.BatchNorm2d(3)
        # self.model_1 = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=0)
        # self.model_2 = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(768, classes, drop_rate, num_bottleneck=768)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            # x1 = self.bn(x1)
            # print(x1.shape)
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            # x2 = self.bn(x2)
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)
        return y1, y2


class Swin(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=True):
        super(Swin, self).__init__()
        self.model_1 = timm.create_model("swin_base_patch4_window12_384", pretrained=True, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("swin_base_patch4_window12_384", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(1024, classes, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            # print(x1.shape)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)
        return y1, y2


class ft_net_LPN(nn.Module):
    def __init__(self, stride=1, init_model=None, pool='avg', block=4):
        super(ft_net_LPN, self).__init__()
        # model_ft = timm.create_model("resnet50", pretrained=True, num_classes=0)
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        self.model = model_ft
        self.model.relu = nn.ReLU(inplace=True)
        self.block = block
        if init_model != None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        # x = self.model(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.shape)
        # print(x.shape)

        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)

        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block), W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block), W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                # print("x", x.shape)
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                # print("x_curr", x_curr.shape)
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    x_pad = functional.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                # print("x_curr", x_curr.shape)
                avgpool = pooling(x_curr)
                # print("pool", avgpool.shape)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = functional.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = functional.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)


class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, pool='avg', share_weight=False, VGG16=False, LPN=False, block=4):
        super(two_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        self.model_1 = ft_net_LPN(pool=pool, block=block)
        # self.model_2 = ft_net_LPN(class_num, stride=stride, pool=pool, block=block)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = ft_net_LPN(pool=pool, block=block)

        if pool == 'avg+max':
            for i in range(self.block):
                name = 'classifier'+str(i)
                setattr(self, name, ClassBlock(4096, class_num, droprate))
        else:
            for i in range(self.block):
                name = 'classifier'+str(i)
                setattr(self, name, ClassBlock(2048, class_num, droprate))

    def forward(self, x1, x2):  # x4 is extra data

        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.part_classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.part_classifier(x2)

        return y1, y2

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            # print(c)
            predict[i] = c(part[i])
            # print(predict[i].shape)
        # print(predict)
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    # import ssl

    # ssl._create_default_https_context = ssl._create_unverified_context
    # model = ViT_two_view_LPN(100, 0.1).cuda()
    # model = Hybird_ViT(100, 0.1).cuda()
    # model = ViT_two_view_LPN(100, 0.1).cuda()
    model = Hybird_ViT(100, 0.1, True).cuda()
    # print(model)
    # model = EfficientNet_b()
    # print(model.device)
    # print(model.extract_features)
    # Here I left a simple forward function.
    # Test the model, before you train it.
    input = torch.randn(1, 3, 384, 384).cuda()
    output1, output2 = model(input, input)
    print(output1.size())
    # print(output)

model_dict = {
    "LPN": two_view_net,
    "resnet": ResNet,
    "seresnet": SEResNet_50,
    "dense": DenseNet,
    "vit": ViT,
    "swin": Swin,
    "hybrid": Hybird_ViT
}
