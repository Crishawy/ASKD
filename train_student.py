from __future__ import print_function

import os
import argparse
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import ConvReg, LinearEmbed

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader, get_dataloader_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, RKDLoss
from distiller_zoo import FSP, RegionalSimilarityLoss, ClassSimilarityLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init
from helper.util import read_cfg


def load_teacher(model_path, model_name, n_cls):
    print('==> loading teacher model')
    model = model_dict[model_name](num_classes=n_cls)
    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)
    print('==> done')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", default="./configs/kd/attention/r32-256_r16-64.yaml", metavar="FILE",
                        help="path to config file", type=str)
    args = parser.parse_args()
    opt = read_cfg(args.cfg)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(opt, batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(opt, batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers,
                                                                is_instance=False)
        n_cls = 100
    elif opt.dataset == 'tinyimagenet':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data, _ = get_dataloader_sample(opt, batch_size=opt.batch_size, num_workers=
            opt.num_workers, is_sample=True, k=opt.nce_k)
        else:
            train_loader, val_loader = get_imagenet_dataloader(opt, batch_size=opt.batch_size,
                                                               num_workers=opt.num_workers)
        n_cls = 200
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, opt.model_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    # config kd
    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())  # 第一阶段训练的layer
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification training
        pass
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'ASKD':
        criterion_kd = nn.ModuleList([])
        criterion_kd.append(ClassSimilarityLoss())
        criterion_kd.append(RegionalSimilarityLoss(opt.r))
        # distillation position
        if opt.group == 'res_group':
            s_channels = [f.shape[1] for f in feat_s[1:-1]]
            t_channels = [f.shape[1] for f in feat_t[1:-1]]
        elif opt.group == 'res_last':
            s_channels = [f.shape[1] for f in feat_s[-2:-1]]
            t_channels = [f.shape[1] for f in feat_t[-2:-1]]
        elif opt.group == 'target':
            s_channels = [f.shape[1] for f in feat_s[opt.layer: opt.layer + 1]]
            t_channels = [f.shape[1] for f in feat_t[opt.layer: opt.layer + 1]]
        else:
            raise NotImplementedError(opt.group)
        print('number of layers:', len(s_channels))
        for i in range(len(s_channels)):
            conv1 = nn.Conv2d(s_channels[i], t_channels[i], 1)
            module_list.append(conv1)
            trainable_list.append(conv1)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        print('Train with gpu..................')
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    best_acc = 0
    best_acc_top5 = 0
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        # train
        time1 = time.time()
        train_acc, train_loss, train_kd_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # test
        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_top5 = test_acc_top5
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)

    print(' * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
          .format(top1=best_acc, top5=best_acc_top5))


if __name__ == '__main__':
    main()
