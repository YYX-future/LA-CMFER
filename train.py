import torch
import torch.nn.functional
from torch.autograd import Variable
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from data import data_prepare
from models import LA-CMFER as models
from pseudo import pseudo
from utils import setup_seed
from utils.utils import inverseDecaySheduler, OptimWithSheduler, OptimizerManager
import numpy as np


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_parse_option():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)

    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--devices', type=str, default='0', help='Set the CUDA_VISIBLE_DEVICES var from this string')


    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l2_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--batch_size', type=int, default=128, help="batch size for single GPU")  # 100
    parser.add_argument('--total_steps', type=int, default=20000, metavar='N', help='total iteration to run')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=100)

    parser.add_argument('--src_domain', type=str, action='append',
                        default=['CK', 'RAF', 'FER2013', 'Aff', 'Oulu', 'JAFFE'])
    parser.add_argument('--tgt_domain', type=str, default="Aff")
    parser.add_argument('--num_classes', type=int, default=7)

    parser.add_argument('--checkpoint_dir', type=str, default=r"../checkpoint")
    parser.add_argument('--last_model_path', type=str, default=None)
    parser.add_argument('--record_folder', type=str, default=r"../records")

    parser.add_argument('--intra', action='store_false')
    parser.add_argument('--global_inter_sample', action='store_false')
    parser.add_argument('--local_inter_sample', action='store_false')
    parser.add_argument('--inter_class', action='store_false')
    parser.add_argument('--inter_baseline', action='store_true')
    parser.add_argument('--hyper', action='store_true')

    parser.add_argument('--l1', action='store_true')
    parser.add_argument('--mse', action='store_true')
    parser.add_argument('--kl', action='store_true')

    parser.add_argument('--hyper_align', type=float, default=0.4, help="hyper parameter of align")
    parser.add_argument('--hyper_inter', type=float, default=0.02, help="hyper parameter of inter")
    parser.add_argument('--hyper_intra', type=float, default=0.5, help="hyper parameter of intra")
    parser.add_argument('--hyper_tgt', type=float, default=0.1, help="hyper parameter of tgt ps")
    parser.add_argument('--threshold', type=float, default=0.4, help="alignment pseudo label threshold")
    parser.add_argument('--ps_threshold', type=float, default=0.9, help="pseudo label threshold")

    args, unparsed = parser.parse_known_args()
    return args


def train(model, target, record_result, record_loss, checkpoint_path):

    setup_seed(args.seed)

    root = r'./data/'

    if args.tgt_domain in ["JAFFE", "CK", "Oulu"]:
        src_loaders, tgt_train_dl, tgt_test_dl = data_prepare.get_loaders_epoch(root, args, True)

    else:
        src_loaders, tgt_train_dl, tgt_test_dl = data_prepare.get_loaders_epoch(root, args, True)

    src_domains = args.src_domain.copy()
    src_domains.remove(args.tgt_domain)

    src_acc, best_tgt_acc, best_iter, start_steps = [[0]] * 2, 0, 0, 0

    if args.last_model_path is not None:
        model_checkpoint = torch.load(args.last_model_path, map_location=device)
        start_steps = model_checkpoint['epoch']

    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
    optimizer_finetune = OptimWithSheduler(
        torch.optim.SGD(model.sharedNet.parameters(), lr=args.lr / 10,
                        weight_decay=args.l2_decay, momentum=args.momentum, nesterov=True),
        scheduler)
    optimizer_add_g = OptimWithSheduler(
        torch.optim.SGD(model.addnetlist_g.parameters(), lr=args.lr,
                        weight_decay=args.l2_decay, momentum=args.momentum, nesterov=True),
        scheduler)
    optimizer_add_l = OptimWithSheduler(
        torch.optim.SGD(model.addnetlist_l.parameters(), lr=args.lr,
                        weight_decay=args.l2_decay, momentum=args.momentum, nesterov=True),
        scheduler)
    optimizer_cls_g = OptimWithSheduler(
        torch.optim.SGD(model.cls_g.parameters(), lr=args.lr,
                        weight_decay=args.l2_decay, momentum=args.momentum, nesterov=True),
        scheduler)
    optimizer_cls_l = OptimWithSheduler(
        torch.optim.SGD(model.cls_l.parameters(), lr=args.lr,
                        weight_decay=args.l2_decay, momentum=args.momentum, nesterov=True),
        scheduler)


    total_steps = tqdm(range(args.total_steps - start_steps), desc='global step')

    epoch_id = 0

    while start_steps < args.total_steps:

        epoch_id += 1
        for it, (((src_data, src_label), d_idx), (tgt_data, tgt_label)) in enumerate(
                zip(src_loaders, tgt_train_dl)):

            model.train()

            p = float(start_steps / args.total_steps)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_data, src_label = src_data.to(device), src_label.to(device)
            tgt_data, tgt_label = tgt_data.to(device), tgt_label.to(device)

            cls_g, cls_l, align_g, align_l, intra, tgt_l, ps_acc_g, ps_acc_l, ps_acc_t\
                = model(args, src_data, tgt_data, src_label, tgt_label, alpha, start_steps)

            loss = cls_g + cls_l + (align_g + align_l) * args.hyper_align + intra * args.hyper_intra \
                   + tgt_l * args.hyper_tgt

            with OptimizerManager([optimizer_finetune, optimizer_add_g, optimizer_add_l, optimizer_cls_g,
                                   optimizer_cls_l]):
                loss.backward()

            start_steps += 1
            total_steps.update()
            model_checkpoint = {
                "net": model.state_dict(),
                "epoch": start_steps
            }
            if start_steps % args.log_interval == 0:
                print(
                    'iter {}: Train target {} total: {:.4f} glob_Loss: {:.4f} local_loss: {:.4f} intra_Loss: {:.4f} '
                    'inter_Loss_g: {:.4f} inter_Loss_l: {:.4f} tgt_Loss_l: {:.4f} '
                    'tgt_lb_acc_g: {:.4f} tgt_lb_acc_l: {:.4f} tgt_lb_acc_t: {:.4f}'.format(
                        start_steps, target, loss.item(), cls_g.item(), cls_l.item(), intra.item() * args.hyper_intra,
                        align_g * args.hyper_align, align_l * args.hyper_align, tgt_l * args.hyper_tgt,
                        ps_acc_g, ps_acc_l, ps_acc_t))

            if start_steps % args.test_interval == 0:
                f1 = open(record_loss, mode='a+')
                f1.writelines(
                    ['loss: ', str(float(loss.item())), ' ',
                     'glob: ', str(float(cls_g.item())), ' ', 'local: ', str(float(cls_l.item())), ' ',
                     'inter_g: ', str(float(align_g * args.hyper_align)), ' ',
                     'inter_l: ', str(float(align_l * args.hyper_align)), ' ',
                     'intra: ', str(float(intra.item() * args.hyper_intra)), ' ',
                     'tgt: ', str(float(tgt_l * args.hyper_tgt)), ' ',
                     ])
                f1.write('\n')
                f1.close()

                f2 = open(record_result, mode='a+')
                test_loss, test_correct, test_acc = test(model, tgt_test_dl, start_steps)

                if max(test_acc) > best_tgt_acc:
                    best_tgt_acc = max(test_acc)
                    best_iter = start_steps
                    best_acc_model_path = '%s/%s_%s_%s.pt' % (
                        checkpoint_path, target, 'best', str(start_steps))
                    torch.save(model_checkpoint, best_acc_model_path)
                f2.writelines(
                    ['best test acc: {:.2f} '.format(float(best_tgt_acc)), ' ',
                     'best iter: ', str(best_iter), ' ',
                     'test acc now: {:.2f} '.format(float(max(test_acc))), ' ',
                     'test acc_global: {:.2f} '.format(float(test_acc[0])), ' ',
                     'test acc_local: {:.2f} '.format(float(test_acc[1])), ' ',
                     'test avg: {:.2f} '.format(float(test_acc[2])), ' ',
                     ])
                f2.write("validation loss：" + str(float(test_loss)) + "\n")

                f2.close()


def test(model, tgt_test_dl, steps):
    model.eval()
    test_loss = 0
    correct = [0] * 3
    correct_confidence = [0] * 3
    cls_loss = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, tgt_label in tgt_test_dl:
            if torch.cuda.is_available():
                data, tgt_label = data.to(device), tgt_label.to(device)
            data, tgt_label = Variable(data), Variable(tgt_label)

            pred_global, pred_local = model(args, data, None, None, tgt_label, None, steps)
            pred_avg = (pred_global + pred_local) / 2

            preds = [pred_global, pred_local, pred_avg]

            pred_temps = [torch.zeros_like(tgt_label) for _ in range(3)]

            for i in range(pred_avg.size(0)):
                temp = torch.cat([p[i] for p in preds], dim=0)
                temp = temp.reshape(-1, pred_avg.size(1))
                for j in range(3):
                    pred_temps[j][i] = pseudo.get_confidence_test_vote(temp)[j]

            test_loss += cls_loss(pred_local, tgt_label).item()

            for j, pred in enumerate(preds):
                pred_label = pred.data.max(1)[1]
                correct[j] += pred_label.eq(tgt_label.data.view_as(pred_label)).cpu().sum()

            for j, pred_temp in enumerate(pred_temps):
                correct_confidence[j] += pred_temp.eq(tgt_label.data.view_as(pred_temp)).cpu().sum()

    test_loss /= len(tgt_test_dl.dataset)

    accs = [100. * c / len(tgt_test_dl.dataset) for c in correct]
    acc_confidences = [100. * c / len(tgt_test_dl.dataset) for c in correct_confidence]

    return test_loss, correct + correct_confidence, accs + acc_confidences


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    global args
    args = get_parse_option()

    for domain in args.src_domain:

        args.tgt_domain = domain

        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        if not os.path.exists(args.record_folder):
            os.mkdir(args.record_folder)

        record_path = os.path.join(args.record_folder, args.tgt_domain)
        if not os.path.exists(record_path):
            os.makedirs(record_path)

        now = datetime.now().strftime("%m-%d-%y_%H%M%S")
        record_result = record_path + "\\" + now + "_" + args.tgt_domain + 'result.txt'
        record_loss = record_path + "\\" + now + "_" + args.tgt_domain + 'loss.txt'
        checkpoint_path = os.path.join(args.checkpoint_dir, args.tgt_domain, now)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        writer = SummaryWriter(os.path.join("../train_logs", args.tgt_domain, now))

        if args.hyper:
            select_txt = os.path.join(os.getcwd(), 'data', 'hp_search', domain + '_hy' + '.txt')
            print("parameter search space: ")
            # parser.add_argument('--hyper_align', type=float, default=0.1, help="hyper parameter of align")
            # parser.add_argument('--hyper_inter', type=float, default=0.02, help="hyper parameter of inter")
            # parser.add_argument('--hyper_intra', type=float, default=0.5, help="hyper parameter of intra")
            # parser.add_argument('--hyper_tgt', type=float, default=0.1, help="hyper parameter of tgt ps")
            # parser.add_argument('--threshold', type=float, default=0.4, help="alignment pseudo label threshold")
            with open(select_txt, 'r') as ff:
                lines = ff.readlines()
            for line in lines:
                hypers = line.strip().split(' ')
                print(hypers)
                # args.intra = str2bool(hypers[0])
                # args.kl = str2bool(hypers[1])
                # args.l1 = str2bool(hypers[2])
                # args.mse = str2bool(hypers[3])
                args.hyper_align = float(hypers[0])
                args.hyper_inter = float(hypers[1])
                args.hyper_intra = float(hypers[2])
                args.hyper_tgt = float(hypers[3])
                args.threshold = float(hypers[4])
                model = models.MDAEFR(args.num_classes).to(device)
                train(model, args.tgt_domain, record_result, record_loss, checkpoint_path)
        else:

            model = models.MDAEFR(args.num_classes).to(device)
            train(model, args.tgt_domain, record_result, record_loss, checkpoint_path)

