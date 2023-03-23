import torch
import math
from vision_transformer_hybrid import _create_vision_transformer_hybrid
from torch import nn
from torch.nn import init, functional
# from utils import get_yaml_value
from resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame, StdConv2d, to_2tuple
from vision_transformer import VisionTransformer, checkpoint_filter_fn, _create_vision_transformer, Block
# from timm.models.vision_transformer_hybrid import _create_vision_transformer_hybrid
from functools import partial
from model_ import weights_init_kaiming, weights_init_classifier
from einops import rearrange
from activation import GeM


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


class Hybird_ViT(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=True):
        super(Hybird_ViT, self).__init__()
        self.block = 2
        conv_layer = partial(StdConv2dSame, eps=1e-8)
        backbone = ResNetV2(
            layers=(3, 4, 9), num_classes=0, global_pool='', in_chans=3,
            preact=False, stem_type="same", conv_layer=conv_layer, act_layer=nn.ReLU)
        model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, num_classes=0)
        model = _create_vision_transformer_hybrid(
            'vit_base_r50_s16_384', backbone=backbone, pretrained=True, **model_kwargs)
        self.model_1 = model
        if share_weight:
            self.model_2 = self.model_1
        # else:
        #     self.model_2 = hybrid_model(layers=(3, 4, 9), img_size=24, patch_size=1, num_classes=1000, depth=12)
        self.classifier_hbp = ClassBlock(2048*3, classes, drop_rate)
        self.classifier_multi = ClassBlock(768*2, classes, drop_rate)
        self.classifier = ClassBlock(768, classes, drop_rate)

        self.proj = nn.Conv2d(768, 1024, kernel_size=1, stride=1)
        self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(1024, 2048, kernel_size=1, bias=False),
                                                 torch.nn.BatchNorm2d(2048),
                                                 torch.nn.ReLU())

        self.bilinear_proj_lpn = torch.nn.Sequential(torch.nn.Conv2d(1024, 2048, kernel_size=1, bias=False),
                                                     torch.nn.BatchNorm2d(2048),
                                                     torch.nn.ReLU())
        self.Vit_block = Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, init_values=None,
                               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               act_layer=nn.GELU)
        self.gem = GeM(1024)
        for m in self.bilinear_proj.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)


        LPN = 1
        if LPN:
            for i in range(self.block):
                # before lpn
                # name = 'classifier' + str(i + 1)
                # after lpn
                name = 'classifier' + str(i)
                setattr(self, name, ClassBlock(1024, classes, drop_rate))
                # print(name)

    def hbp(self, conv1, conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj(conv1)
        proj_2 = self.bilinear_proj(conv2)

        X = proj_1 * proj_2
        # print(X.shape)
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        # print(X.shape)
        X = X.view(N, 2048)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def restore_vit_feature(self, x):
        x = x[:, 1:, :]
        x = rearrange(x, "b (h w) y -> b y h w", h=24, w=24)
        x = self.proj(x)
        return x

    def fusion_features(self, x, t, model):

        y = []
        # with torch.no_grad():
        x, p_f, v_f, l_f = model(x, t)

        l_f = self.Vit_block(l_f)

        # direct softmax
        y0, f = self.classifier(x)

        # multi modal softmax
        v_f = self.restore_vit_feature(v_f)
        l_f = self.restore_vit_feature(l_f)

        # HBP softmax 3 layer feature X multiply
        x1 = self.hbp(p_f, v_f)
        x2 = self.hbp(p_f, l_f)
        x3 = self.hbp(v_f, l_f)
        x = torch.concat([x1, x2, x3], dim=1)

        y2, hbp_f = self.classifier_hbp(x)

        result = self.get_part_pool(v_f)

        if self.training:
            y3, lpn_f = self.part_classifier(result)
        else:
            lpn_f = self.part_classifier(result)
            y3 = [None, None]

        y.append(y0)
        # y.append(y1)
        y.append(y2)
        y.append(y3[0])
        y.append(y3[1])
        # y.append(y4)
        if self.training:
            f_all = torch.concat([f, hbp_f, lpn_f], dim=1)
        else:
            f = f.view(f.size()[0], f.size()[1], 1)
            hbp_f = hbp_f.view(hbp_f.size()[0], hbp_f.size()[1], 1)
            f_all = torch.concat([f, hbp_f, lpn_f], dim=2)
        return y, f_all

    def forward(self, x1, x2, t1, t2):

        if x1 is None:
            y1 = None
            f1 = None
            t1 = None
            output1 = None
        else:
            y1, f1 = self.fusion_features(x1, t1, self.model_1)

        if x2 is None:
            y2 = None
            f2 = None
            t2 = None
            output2 = None
        else:
            y2, f2 = self.fusion_features(x2, t2, self.model_2)

        if self.training:
            return y1, y2, f1, f2
            # output1, output2
        else:
            # print("ff12", f2.shape)
            return f1, f2

    def get_part_pool(self, x, pool='max', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                # print("x", x.shape)
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
                # print("x_curr", x_curr.shape)
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = functional.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                # print("x_curr", x_curr.shape)
                avgpool = pooling(x_curr)
                # print("pool", avgpool.shape)
                result.append(avgpool)
                # print(x_curr.shape)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = functional.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = functional.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
                # print(x.shape)
        return torch.concat(result, dim=2)

    def part_classifier(self, x):
        part = {}
        predict = {}
        features = []
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)

            name = 'classifier' + str(i)
            c = getattr(self, name)
            # print(c)
            predict[i], feature = c(part[i])
            features.append(feature)

            # print(predict[i][0].shape)
        # print(predict)
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y, torch.concat(features, dim=1)


if __name__ == '__main__':
    # create_model()
    model = Hybird_ViT(classes=701, drop_rate=0.3).cuda()

    feature = torch.randn(8, 3, 384, 384).cuda()
    text = torch.rand(8, 1, 768).cuda()
    output = model(feature, feature, text, text)
    print(output)
