#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    # source_size = int(source.size()[0]) if int(source.size()[0]) < 255 else int(len(source.size()))
    # target_size = int(target.size()[0]) if int(target.size()[0]) < 255 else int(len(target.size()))
    source_size = int(source.size()[0])
    target_size = int(target.size()[0])
    n_samples = source_size + target_size
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]

    return XX, YY, 2 * XY


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    # temp = XX
    # temp = torch.div(temp, n)
    # temp = torch.div(temp, n).sum(dim=-1).view(1, -1)
    # print(temp)
    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss，Source<->Source
    # print(XX)

    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()

    return loss


def get_weight(out1, out2, label, pred_label, mask_src, mask_tgt, num_classes):

    # pred_label = pred.data.max(1)[1]
    weight_source = torch.zeros_like(label).float()
    weight_target = torch.zeros_like(pred_label).float()

    for i in range(label.shape[0]):
        equal = torch.eq(pred_label, label[i]).int()
        num = equal.sum()
        equal = equal.unsqueeze(dim=1)
        buff = torch.mul(out2, equal)
        data = out1[i].unsqueeze(dim=0).expand(buff.size())
        if num != 0:
            buff = (torch.nn.functional.one_hot(label[i].to(torch.int64), num_classes) * out1[i].max(0)[0]).to(device)
            weight_source[i] = torch.square(data - buff).mean() * mask_src[i]
        else:
            buff = (torch.nn.functional.one_hot(label[i].to(torch.int64), num_classes) * out1[i].max(0)[0]).to(device)
            weight_source[i] = torch.square(data - buff).mean() * mask_src[i]

    for i in range(label.shape[0]):
        equal = torch.eq(label, pred_label[i]).int()
        num = equal.sum()
        equal = equal.unsqueeze(dim=1)
        buff = torch.mul(out1, equal)
        data = out2[i].unsqueeze(dim=0).expand(buff.size())
        if num != 0:
            buff = (torch.nn.functional.one_hot(pred_label[i].to(torch.int64), num_classes) * out2[i].max(0)[0]).to(device)
            weight_target[i] = torch.square(data - buff).mean() * mask_tgt[i]
        else:
            buff = (torch.nn.functional.one_hot(pred_label[i].to(torch.int64), num_classes) * out2[i].max(0)[0]).to(device)
            weight_target[i] = torch.square(data - buff).mean() * mask_tgt[i]

    if weight_source.sum() != 0:
        weight_source = weight_source / weight_source.sum()
    else:
        weight_source = torch.zeros(weight_source.size()).to(device)

    if weight_target.sum() != 0:
        weight_target = weight_target / weight_target.sum()
    else:
        weight_target = torch.zeros(weight_target.size()).to(device)

    weight_source = weight_source.unsqueeze(dim=1)
    weight_target = weight_target.unsqueeze(dim=1)
    weight_XX = torch.mul(weight_source, weight_source.t())
    weight_YY = torch.mul(weight_target, weight_target.t())
    weight_XY = torch.mul(weight_source, weight_target.t())

    return weight_XX, weight_YY, weight_XY


def get_cluster_loss(out1, out2, label, pred, mask_src, mask_tgt):

    # pred = pred.data.max(1)[1]
    label_set = []
    loss = 0
    intra_dis = 0
    inter_dis = 0
    num_intra = 0
    num_inter = 0
    for i in range(0, label.shape[0]):
        if (label[i] not in label_set) and mask_src[i] != 0:
            label_set.append(label[i])
        if (pred[i] not in label_set) and mask_tgt[i] != 0:
            label_set.append(pred[i])
    for i in range(0, len(label_set)):
        equal_src = torch.eq(label, label_set[i]).int()
        equal_tgt = torch.eq(pred, label_set[i]).int()
        if equal_src.sum() != 0 and equal_tgt.sum() != 0:
            num_intra += 1
            index = equal_src.nonzero(as_tuple=False)
            index = index.squeeze(dim=1)
            data_src = out1[index]
            index = equal_tgt.nonzero(as_tuple=False)
            index = index.squeeze(dim=1)
            data_tgt = out2[index]
            intra_dis += mmd_loss(data_src, data_tgt)

    for i in range(0, len(label_set)):
        for j in range(0, len(label_set)):
            if i != j:
                equal_src = torch.eq(label, label_set[i]).int()
                equal_tgt = torch.eq(pred, label_set[j]).int()
                if equal_src.sum() != 0 and equal_tgt.sum() != 0:
                    num_inter += 1
                    index = equal_src.nonzero(as_tuple=False)
                    index = index.squeeze(dim=1)
                    data_src = out1[index]
                    index = equal_tgt.nonzero(as_tuple=False)
                    index = index.squeeze(dim=1)
                    data_tgt = out2[index]
                    inter_dis += mmd_loss(data_src, data_tgt)

    if num_inter != 0 and num_intra != 0:
        loss += intra_dis / num_intra - (inter_dis / num_inter)
    return loss





