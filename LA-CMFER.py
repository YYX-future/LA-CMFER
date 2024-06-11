import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import loss.smooth_cls_1 as smooth_cls
from loss.mmd_1 import mmd, mmd_loss, get_weight, get_cluster_loss
from loss.intra_1 import contras_cls
from pseudo.pseudo import get_ps_label_acc, select_pseudo_labels
from fightingcv_attention.attention.CBAM import CBAMBlock

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_path_resnet = {
    'resnet18': r"../pretrained_models/resnet18-5c106cde.pth",
    'resnet50': r"../pretrained_models/resnet50-19c8e357.pth",

}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)  #
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  #
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  #
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.3)

        # 　权值参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        return x


class AttentionBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAMBlock(planes, 16)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ADDNET(nn.Module):
    def __init__(self, AttentionBlock):
        super(ADDNET, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Shared feature extraction module.
        self.layer3 = self._make_layer(AttentionBlock, 128, 256, 6, stride=2)  # 14x14x256
        self.layer4 = self._make_layer(AttentionBlock, 256, 1024, 3, stride=2)  # 7x7x512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Global.
        out_g = self.layer3(x)  # 14x14x256
        out_g = self.layer4(out_g)  # 7x7x512
        out_g = self.avgpool(out_g)
        out_g = torch.flatten(out_g, 1)
        return out_g


class CropNet(nn.Module):
    def __init__(self, BasicBlock):
        super(CropNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Shared feature extraction module.
        self.layer5_1 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)  # 7x7x128
        self.layer5_2 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)  # 7x7x128
        self.layer5_3 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)  # 7x7x128
        self.layer5_4 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)  # 7x7x128
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Local.
        patch_11 = x[:, :, 0:14, 0:14]
        patch_12 = x[:, :, 0:14, 14:28]
        patch_21 = x[:, :, 14:28, 0:14]
        patch_22 = x[:, :, 14:28, 14:28]
        out_l11 = self.layer5_1(patch_11)
        out_l12 = self.layer5_2(patch_12)
        out_l21 = self.layer5_1(patch_21)
        out_l22 = self.layer5_2(patch_22)
        out_l11 = self.avgpool(out_l11)
        out_l11 = torch.flatten(out_l11, 1)
        out_l12 = self.avgpool(out_l12)
        out_l12 = torch.flatten(out_l12, 1)
        out_l21 = self.avgpool(out_l21)
        out_l21 = torch.flatten(out_l21, 1)
        out_l22 = self.avgpool(out_l22)
        out_l22 = torch.flatten(out_l22, 1)
        out = torch.cat([out_l11, out_l12, out_l21, out_l22], dim=1)
        return out


def resnet18(pretrained=False, **kwargs):

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_path_resnet['resnet18'], map_location='cpu'))
    # pretrained_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return model


class MDAEFR(nn.Module):
    def __init__(self, num_classes=7):
        super(MDAEFR, self).__init__()
        self.num_classes = num_classes
        self.num_domains = 5
        self.gfeat = 1024  # 512
        self.lfeat = 1024
        # Domain-shared
        self.sharedNet = resnet18(True)
        # global features
        self.addnetlist_g = ADDNET(AttentionBlock)
        self.cls_g = nn.Linear(self.gfeat, self.num_classes)
        # local fine-grained features
        self.addnetlist_l = CropNet(AttentionBlock)
        self.cls_l = nn.Linear(self.lfeat, self.num_classes)
        # fusion
        # self.classifier_f = nn.Linear(1024, self.num_classes)
        # cls_Loss
        self.l_smooth = smooth_cls.CrossEntropyLabelSmooth(self.num_classes).to(device)

    def forward(self, args, src_data, tgt_data, src_label, tgt_label):

        align_g, align_l, dis_l, intra, adr_g, adr_l = 0, 0, 0, 0, 0, 0

        if self.training:

            src_data = self.sharedNet(src_data)
            tgt_data = self.sharedNet(tgt_data)

            src_fea_g = self.addnetlist_g(src_data)
            src_pre_g = self.cls_g(src_fea_g)
            tgt_fea_g = self.addnetlist_g(tgt_data)
            tgt_pre_g = self.cls_g(tgt_fea_g)

            src_fea_l = self.addnetlist_l(src_data)
            src_pre_l = self.cls_l(src_fea_l)
            tgt_fea_l = self.addnetlist_l(tgt_data)
            tgt_pre_l = self.cls_l(tgt_fea_l)

            __, __, mask_src_g, acc_g = get_ps_label_acc(src_pre_g, args.threshold, src_label)
            __, __, mask_src_l, acc_l = get_ps_label_acc(src_pre_l, args.threshold, src_label)
            prob_tgt_g, ps_lb_tgt_g, mask_tgt_g, ps_acc_g = get_ps_label_acc(tgt_pre_g, args.threshold, tgt_label)
            prob_tgt_l, ps_lb_tgt_l, mask_tgt_l, ps_acc_l = get_ps_label_acc(tgt_pre_l, args.threshold, tgt_label)

            tgt_l, ps_acc_t = select_pseudo_labels(tgt_pre_g, tgt_pre_l, args.ps_threshold, tgt_label)

            if args.intra:
                intra += contras_cls(tgt_pre_g, tgt_pre_l)

            # if args.l1:
            #     intra += F.l1_loss(tgt_pre_g, tgt_pre_l)
            #
            # if args.mse:
            #     intra += F.mse_loss(tgt_pre_g, tgt_pre_l)
            #
            # if args.kl:
            #     pred1_log_softmax = F.log_softmax(tgt_pre_g, dim=1)
            #     pred2_softmax = F.softmax(tgt_pre_l, dim=1)
            #     intra += 0.5 * F.kl_div(pred1_log_softmax, pred2_softmax, reduction='batchmean')
            #     pred2_log_softmax = F.log_softmax(tgt_pre_l, dim=1)
            #     pred1_softmax = F.softmax(tgt_pre_g, dim=1)
            #     intra += 0.5 * F.kl_div(pred2_log_softmax, pred1_softmax, reduction='batchmean')

            if args.global_inter_sample:

                weight_XX, weight_YY, weight_XY = get_weight(src_pre_g, tgt_pre_g, src_label, ps_lb_tgt_g,
                                                             mask_src_g, mask_tgt_g, args.num_classes)

                XX, YY, XY = mmd(src_fea_g, tgt_fea_g)

                align_g = ((torch.mul(weight_XX, XX) + torch.mul(weight_YY, YY) - torch.mul(weight_XY, XY)).sum())

                if args.inter_class:
                    align_g += get_cluster_loss(src_fea_g, tgt_fea_g, src_label, ps_lb_tgt_g, mask_src_g, mask_tgt_g) \
                               * args.hyper_inter

            elif args.inter_baseline:
                align_g = mmd_loss(src_fea_g, tgt_fea_g)

            # local
            if args.local_inter_sample:

                weight_XX, weight_YY, weight_XY = get_weight(src_pre_l, tgt_pre_l, src_label, ps_lb_tgt_l,
                                                             mask_src_l, mask_tgt_l, args.num_classes)

                XX, YY, XY = mmd(src_fea_l, tgt_fea_l)

                align_l = ((torch.mul(weight_XX, XX) + torch.mul(weight_YY, YY) - torch.mul(weight_XY, XY)).sum())

                if args.inter_class:
                    align_l += get_cluster_loss(src_fea_l, tgt_fea_l, src_label, ps_lb_tgt_l, mask_src_l, mask_tgt_l) \
                               * args.hyper_inter

            elif args.inter_baseline:
                align_l = mmd_loss(src_fea_l, tgt_fea_l)

            cls_g = self.l_smooth(src_pre_g, src_label)
            cls_l = self.l_smooth(src_pre_l, src_label)

            return cls_g, cls_l, align_g, align_l, intra, tgt_l, ps_acc_g, ps_acc_l, ps_acc_t

        else:
            data = self.sharedNet(src_data)
            tgt_fea_g = self.addnetlist_g(data)
            tgt_pre_g = self.cls_g(tgt_fea_g)
            tgt_fea_l = self.addnetlist_l(data)
            tgt_pre_l = self.cls_l(tgt_fea_l)

            return tgt_pre_g, tgt_pre_l


