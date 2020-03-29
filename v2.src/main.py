#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:53:51 2019

@author: binary
"""
import sys
sys.path.append("../")

import os
import time
import torch.nn as nn
import torch
import fire
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

import loader
from base import libiou
from base.snet import s_resnet
from base.dnet53 import darknet53_v2 as darknet53
from base import cross_entropy
from base import libnet
from base import util
import conf

def build_target(rs, cents, objs, anchors_cuda):
    grid = rs.shape[1]
    _boxes = rs[:, :, :, :, conf.loc_box:conf.loc_box+4]
    target_p = torch.zeros(rs.shape[:-1])
    target_box = torch.zeros(list(rs.shape[:-1])+[4])
    target_cls = torch.zeros(rs.shape[:-1])
    mask = torch.zeros(rs.shape[:-1], dtype=torch.bool)

    for bat in range(rs.shape[0]):
        for i, cent in enumerate(cents[bat][0]):
            # iou anchor
            copy_anchors = torch.zeros(len(anchors_cuda), 4)
            copy_anchors[:, :2] = cent[:2]
            copy_anchors[:, 2:] = anchors_cuda
            copy_anchors_rec = libiou.cent2rec_anchor(copy_anchors)

            copy_wh = torch.zeros(4)
            copy_wh[:2] = cent[:2]
            copy_wh[2:] = cent[2:] * grid
            copy_wh_rec = libiou.cent2rec_anchor(copy_wh.unsqueeze(0)).squeeze(0)

            iou_anchor = libiou.cpu_iou_allbyall(copy_anchors_rec, copy_wh_rec)[0]
            _, best_anchor_ind = iou_anchor.max(dim=-1)

            cell_x, cell_y = cents[bat][1][i]
            select_point = _boxes[bat, cell_y, cell_x, best_anchor_ind]
            mask[bat, cell_y, cell_x, best_anchor_ind] = 1

            lb = conf.loc_box
            rs_box = select_point.clone().detach()
            rs_box[lb+2:lb+4] = torch.exp(rs_box[lb+2:lb+4]) * anchors_cuda[best_anchor_ind]

            dest_rec_iouonly = libiou.cent2rec_samecell(cent.unsqueeze(0), grid).squeeze(0)
            bbox = libiou.cent2rec_anchor(rs_box.unsqueeze(0))[0]
            max_iou_val = libiou.cpu_iou(bbox, dest_rec_iouonly.detach())[0]

            # p
            target_p[bat, cell_y, cell_x, best_anchor_ind] = max_iou_val

            # box
            target_box[bat, cell_y, cell_x, best_anchor_ind, :2] = cent[:2]
            target_box[bat, cell_y, cell_x, best_anchor_ind, 2:] = torch.log(cent[2:] * grid / conf.anchors[best_anchor_ind])

            # class
            target_cls[bat, cell_y, cell_x, best_anchor_ind] = objs[bat][i]

    return target_p, target_box, target_cls.long(), mask

def train(net, optim, imgx, cents, objs, device):
    if optim is None:
        net.eval()
    else:
        net.train()

    rs = net(imgx.to(device))
    rs = libnet.refine(rs, conf.class_num+5)

    lossmap = {conf.gk_obj:[], conf.gk_noobj:[], conf.gk_coor:[], conf.gk_class:[]}
    t_p, t_box, t_cls, mask = build_target(rs, cents, objs, conf.anchors.to(device))

    lose_fun = nn.MSELoss(reduction=conf.loss_method)
    lose_fun_cross = cross_entropy.sCrossEntropyLoss(reduction=conf.loss_method)
    bat = rs.shape[0] if conf.loss_method == 'sum' else 1

    nomask = ~mask
    rs_obj = rs[mask]
    t_p = t_p.to(device)

    n_tp = t_p[mask].max(torch.zeros_like(t_p[mask]) + 0.9)
    lossmap[conf.gk_obj] = lose_fun(rs_obj[:, conf.loc_p], n_tp) / bat

    lossmap[conf.gk_noobj] = lose_fun(rs[nomask][:, conf.loc_p], t_p[nomask]) / bat

    lossmap[conf.gk_coor] = lose_fun(rs_obj[:, conf.loc_box:conf.loc_box+4], t_box[mask].to(device)) / bat

    lossmap[conf.gk_class] = lose_fun_cross(rs_obj[:, conf.loc_cls:conf.loc_cls+conf.class_num], t_cls[mask].to(device)) / bat

    losses = []
    for k, loss in lossmap.iteritems():
        losses.append(conf.gmap[k] * loss)

    losses = torch.stack(losses).sum()

    if optim is not None:
        optim.zero_grad()
        losses.backward()
        optim.step()

    lossmap['iou'] = t_p[mask].detach().cpu().mean()
    return losses, lossmap

def adjust_eta(optimizer, rate):
    for param in optimizer.param_groups:
        param['lr'] *= rate

def save_weight(net, codename, models, bestloss, force=False):
    dpath = './checkpoint'

    rname = '%s_%s' % (codename, models) if codename else models
    dumpname = 'v2.%s_bestloss_%.1f' % (rname, bestloss)
    savefile = '%s/%s' % (dpath, dumpname)
    linkname = '%s/v2.last_%s' % (dpath, rname)
    if not force and os.path.exists(linkname):
        mtime = os.stat(linkname).st_mtime
        diff = time.time() - mtime
        if diff < 60:
            return diff

    torch.save(net.state_dict(), savefile)
    try:
        os.remove(linkname)
    except Exception:
        pass

    return os.symlink(dumpname, linkname)

def train_warp(net, optimizer, it_loader, device, name, pretitle, writer, ep):
    checklosses = []
    lossmap_w = {}
    for iterat, (_, imgx, cents, objs, _) in enumerate(it_loader):
        gind = iterat % len(conf.gmode)
        sgrid = conf.gmode[gind]
        it_loader.dataset.grid = sgrid
        it_loader.dataset.isize = sgrid * 32

        _, lossmap = train(net, optimizer, imgx, cents, objs, device)
        losmap_sum = sum(lossmap.values())
        checklosses.append(losmap_sum.detach().cpu())
        for k, v in lossmap.iteritems():
            if k not in lossmap_w:
                lossmap_w[k] = []
            lossmap_w[k].append(v.detach().cpu())

    checkloss_mean = torch.stack(checklosses).mean()

    if writer:
        if optimizer:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('%s_%s/eta' % (name, pretitle), lr, global_step=ep)

        writer.add_scalar('%s_%s/losses' % (name, pretitle), checkloss_mean, global_step=ep)
        for k, vs in lossmap_w.iteritems():
            writer.add_scalar('%s_%s/%s' % (name, pretitle, k), torch.stack(vs).mean(), global_step=ep)

    return float(checkloss_mean)

def lrfun(ep):
    eta_decay = conf.eta_decay
    warmup, step1, step2 = sorted(eta_decay.keys())
    if ep <= warmup:
        return eta_decay[warmup]

    x = 1.0
    if ep > step1:
        x = eta_decay[step1]

    if ep > step2:
        x = eta_decay[step2]

    return x

def main(epoch, eta=0.001, lamb=1e-4, gmomentum=0.9, bat=32, method_optim='sgd', checkpoint=None, cuda='cuda', codename=None, models=None, jitterimg=True, shift=True, maxsize=0, summary=True):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    if not models:
        models = util.extact_model(checkpoint)

    if models == 'darknet53':
        net = darknet53(conf.class_num)
    else:
        net = s_resnet(models, True, conf.class_num, 'v2')

    net = net.to(device)

    SGD = method_optim.lower() == 'sgd'

    if SGD:
        optimizer = torch.optim.SGD(net.parameters(), lr=eta, momentum=gmomentum, weight_decay=lamb)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=eta, weight_decay=lamb)

    print 'device:', device, torch.cuda.get_device_name(device) if torch.cuda.is_available() and cuda != 'cpu' else 'cpu'
    print 'models:', models
    print 'jitterimg', jitterimg
    print 'shift', shift
    print 'optim:', 'SGD' if SGD else 'ADAM'

    if checkpoint:
        print 'loading checkpoint', checkpoint
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    print 'init trainset...'
    trainset = loader.imgloader(conf.trainroot, conf.v_resize, grid=conf.v_resize/32, jitterimg=jitterimg, shift=shift, maxsize=maxsize, cache_ori=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bat, shuffle=True, collate_fn=loader.collate, num_workers=1)
    print 'train_size:%s' % len(trainset)

    print 'init validset...'
    validset = loader.imgloader(conf.validroot, conf.v_resize, grid=conf.v_resize/32, jitterimg=False, shift=False, maxsize=maxsize, cache_ori=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=bat, shuffle=True, collate_fn=loader.collate, num_workers=1)
    print 'valid_size:%s' % len(validset)

    pretitle = models.split('.')[0]
    writer = None

    lr_optim = torch.optim.lr_scheduler.LambdaLR(optimizer, lrfun)

    bestloss = 8.0
    checkloss = 0
    Trange = trange(1, epoch + 1)
    for ep in Trange:
        if summary and writer is None:
            writer = SummaryWriter()

        # train
        trainloss = train_warp(net, optimizer, trainloader, device, 'train', pretitle, writer, ep)

        # valid
        with torch.no_grad():
            checkloss = train_warp(net, None, validloader, device, 'valid', pretitle, writer, ep)
            if checkloss < bestloss:
                bestloss = checkloss
                save_weight(net, codename, models, bestloss)

        lr_optim.step()

        desc = 'ETA:%.5f tL:%.1f cL:%.1f' % (optimizer.param_groups[0]['lr'], trainloss, checkloss)
        Trange.set_description_str(desc)

    save_weight(net, '%s.end%s' % (codename, ep), models, checkloss, True)

if __name__ == "__main__":
    fire.Fire(main)
