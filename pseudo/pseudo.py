import torch
import torch.nn.functional as F


def get_ps_label_acc(pred, threshold, label, t=1):

    logit = torch.softmax(pred / t, dim=1)
    max_probs, label_p = torch.max(logit, dim=1)
    mask = max_probs.ge(threshold).float()

    right_labels = (label_p == label).float() * mask
    ps_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

    return logit, label_p, mask, ps_label_acc


def get_ps_label_only(pred, threshold, t=1):

    logit = torch.softmax(pred / t, dim=1)
    max_probs, label_p = torch.max(logit, dim=1)
    mask = max_probs.ge(threshold).float()

    return logit, label_p, mask


def get_confidence_test(pred):
    # pred = F.softmax(pred, dim=-1)
    index_class = 0
    max_confidence = 0
    for i in range(pred.size(0)):
        # print(pred[i])
        # data = (pred[i].data.max(-1)[-1])
        # print(data)
        index = pred[i].argmax(dim=-1)
        # print(index)
        # print(pred[i][index])
        # print(pred[i][data])
        if pred[i][index] > max_confidence:
            max_confidence = pred[i][index]
            index_class = index
    return index_class


def get_confidence_test_vote(pred):
    index_class = 0
    max_confidence = 0
    index_vote_list = []
    for i in range(pred.size(0)):
        index = pred[i].argmax(dim=-1)
        index_vote_list.append(index)
        if pred[i][index] > max_confidence:
            max_confidence = pred[i][index]
            index_class = index
    vote_index, count = max_list(index_vote_list)
    if count == 1 or count == 3:
        return index_class, vote_index, index_class
    else:
        return index_class, vote_index, vote_index


def get_entropy(pred):
    # pred = F.softmax(pred, dim=-1)
    entropy = (-pred * torch.log(pred)).sum(-1)
    return entropy


def get_confidence(pred):
    pred = F.softmax(pred, dim=-1)
    # print(pred.size())
    index = pred.argmax(dim=-1)
    # print(index)
    # print(pred[index])
    return pred[index]


def max_list(lt):
    temp = 0
    max_ele = 0
    for i in lt:
        if lt.count(i) >= temp:
            max_ele = i
            temp = lt.count(i)
    return max_ele, temp


def select_pseudo_labels(pred1, pred2, threshold, tgt_label):

    prob1 = F.softmax(pred1, dim=-1)
    prob2 = F.softmax(pred2, dim=-1)
    prob1, pred1_classes = torch.max(prob1, dim=1)
    prob2, pred2_classes = torch.max(prob2, dim=1)

    same_prediction = (pred1_classes == pred2_classes)

    high_confidence = (prob1 > threshold) | (prob2 > threshold)
    pseudo_labels = pred1_classes

    high_mask = same_prediction & high_confidence

    # higher_confidence_mask = prob1 < prob2
    # pseudo_labels[~high_mask & higher_confidence_mask] = pred2_classes[~high_mask & higher_confidence_mask]

    loss_1 = (F.cross_entropy(pred1, pseudo_labels, reduction="none") * high_mask).sum() / high_mask.sum()
    loss_2 = (F.cross_entropy(pred2, pseudo_labels, reduction="none") * high_mask).sum() / high_mask.sum()
    total_loss = loss_1 + loss_2

    if torch.isnan(total_loss):
        total_loss = 0

    right_labels = (pseudo_labels == tgt_label).float() * high_mask
    ps_acc_t = right_labels.sum() / max(high_mask.sum(), 1.0)

    return total_loss, ps_acc_t


def select_pseudo_labels_three(pred1, pred2, pred3, threshold, tgt_label):

    prob1 = F.softmax(pred1, dim=-1)
    prob2 = F.softmax(pred2, dim=-1)
    prob1, pred1_classes = torch.max(prob1, dim=1)
    prob2, pred2_classes = torch.max(prob2, dim=1)

    same_prediction = (pred1_classes == pred2_classes)

    high_confidence = (prob1 > threshold) | (prob2 > threshold)

    pseudo_labels = pred1_classes

    high_mask = same_prediction & high_confidence

    loss_1 = (F.cross_entropy(pred1, pseudo_labels, reduction="none") * high_mask).sum() / high_mask.sum()
    loss_2 = (F.cross_entropy(pred2, pseudo_labels, reduction="none") * high_mask).sum() / high_mask.sum()
    total_loss = loss_1 + loss_2

    if torch.isnan(total_loss):
        total_loss = 0

    right_labels = (pseudo_labels == tgt_label).float() * high_mask
    ps_acc_t = right_labels.sum() / max(high_mask.sum(), 1.0)

    return total_loss, ps_acc_t


def get_tgt_l(tgt_pre_f, threshold, label):

    prob = F.softmax(tgt_pre_f, dim=-1)
    prob, pseudo_labels = torch.max(prob, dim=1)
    mask = prob.ge(threshold).float()
    tgt_l = (F.cross_entropy(tgt_pre_f, pseudo_labels, reduction="none") * mask).sum() / mask.sum()
    if torch.isnan(tgt_l):
        tgt_l = 0
    right_labels = (pseudo_labels == label).float() * mask
    ps_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

    return tgt_l, ps_label_acc
