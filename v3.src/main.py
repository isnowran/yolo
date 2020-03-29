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
import numpy as np

import loader
from base import libiou
from base import util
from base.dnet53 import darknet53_v3 as darknet53
from base.snet import s_resnet
from base import cross_entropy
from base import libnet
import conf

def build_target(rss, cinfo, objs, device):
    boxes = [rs[:, :, :, :, conf.loc_box:conf.loc_box+4] for rs in rss]
    target_ps = [torch.zeros(rs.shape[:-1]) for rs in rss]
    target_boxes = [torch.zeros(list(rs.shape[:-1])+[4]) for rs in rss]
    target_clss = [torch.zeros(rs.shape[:-1], dtype=torch.long) for rs in rss]
    masks = [torch.zeros(rs.shape[:-1], dtype=torch.bool) for rs in rss]

    for bat in range(rss[0].shape[0]):
        for i, (cent, cellID, best_rsind, best_anchorind) in enumerate(cinfo[bat]):
            cent = cent.to(device)
            grid = rss[best_rsind].shape[1]

            cell_x, cell_y = cellID
            select_point = boxes[best_rsind][bat, cell_y, cell_x, best_anchorind]
            select_anchor = conf.anchor_split[best_rsind][best_anchorind].to(device)

            # mask
            masks[best_rsind][bat, cell_y, cell_x, best_anchorind] = 1

            lb = conf.loc_box
            rs_box = select_point.clone().detach()
            rs_box[lb+2:lb+4] = torch.exp(rs_box[lb+2:lb+4]) * select_anchor
            bbox = libiou.cent2rec_anchor(rs_box.unsqueeze(0))[0]

            dest_rec_iouonly = libiou.cent2rec_samecell(cent.unsqueeze(0), grid).squeeze(0)
            max_iou_val = libiou.cpu_iou(bbox, dest_rec_iouonly.detach())[0]

            # p
            target_ps[best_rsind][bat, cell_y, cell_x, best_anchorind] = max_iou_val

            # box
            target_boxes[best_rsind][bat, cell_y, cell_x, best_anchorind, :2] = cent[:2]
            target_boxes[best_rsind][bat, cell_y, cell_x, best_anchorind, 2:] = torch.log(cent[2:] * grid / select_anchor)

            # class
            target_clss[best_rsind][bat, cell_y, cell_x, best_anchorind] = objs[bat][i]

    return target_ps, target_boxes, target_clss, masks

def train(net, optim, imgx, cinfo, objs, device):
    if optim is None:
        net.eval()
    else:
        net.train()

    loss_method = 'sum'

    loss_fun_focal = cross_entropy.BCE_focal_loss(reduction=loss_method)
    loss_fun = nn.MSELoss(reduction=loss_method)
    loss_fun_cross = cross_entropy.sCrossEntropyLoss(reduction=loss_method)

    rss = net(imgx.to(device))
    rss = [libnet.refine(r, conf.class_num+5) for r in rss]
    t_ps, t_boxes, t_clss, masks = build_target(rss, cinfo, objs, device)

    ious = torch.zeros(3) - 1
    lossmapes = [{conf.gk_obj:None, conf.gk_noobj:None, conf.gk_coor:None, conf.gk_class:None} for _ in range(len(rss))]

    gloss = {}
    for i, rs in enumerate(rss):
        bat = rs.shape[0] if loss_method == 'sum' else 1
        t_p = t_ps[i].to(device)

        mask = masks[i]
        nomask = ~mask
        lossmap = lossmapes[i]

        lossmap[conf.gk_noobj] = loss_fun_focal(rs[nomask][:, conf.loc_p], t_p[nomask]) / bat
        if mask.any():
            rs_obj = rs[mask]
            t_box = t_boxes[i]
            t_cls = t_clss[i]

            ious[i] = t_p[mask].detach().cpu().mean()

            n_tp = t_p[mask].max(torch.zeros_like(t_p[mask]) + 0.95)
            lossmap[conf.gk_obj] = loss_fun_focal(rs_obj[:, conf.loc_p], n_tp) / bat
            lossmap[conf.gk_coor] = loss_fun(rs_obj[:, conf.loc_box:conf.loc_box+4], t_box[mask].to(device)) / bat
            lossmap[conf.gk_class] = loss_fun_cross(rs_obj[:, conf.loc_cls:conf.loc_cls+conf.class_num], t_cls[mask].to(device)) / bat

        for k, loss in lossmap.items():
            if k not in gloss:
                gloss[k] = []

            gloss[k].append(loss.detach().cpu() if loss else torch.tensor(0.0))

    if optim is not None:
        losses = []

        for it, lossmap in enumerate(lossmapes):
            for k, loss in lossmap.iteritems():
                if not loss:
                    continue

                gk = torch.stack(gloss[k])
                g = gk / gk.sum()
                losses.append(conf.gmap[k] * loss * g[it])

        losses = torch.stack(losses).sum()
        optim.zero_grad()
        losses.backward()
        optim.step()

    for i, iou in enumerate(ious):
        lossmapes[i]['ious'] = iou

    return lossmapes

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

def save_weight(net, codename, models, bestloss, force=False):
    dpath = './checkpoint'

    rname = '%s_%s' % (codename, models) if codename else models
    dumpname = 'v3.%s_bestloss_%.1f' % (rname, bestloss)
    savefile = '%s/%s' % (dpath, dumpname)
    linkname = '%s/v3.last_%s' % (dpath, rname)
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
    lossmap_write = [{}, {}, {}]
    for _, imgx, cinfo, objs, _ in it_loader:
        lossmapes = train(net, optimizer, imgx, cinfo, objs, device)

        if writer:
            for i, lossmap in enumerate(lossmapes):
                for k, v in lossmap.iteritems():
                    if not v:
                        continue

                    if k not in lossmap_write[i]:
                        lossmap_write[i][k] = []

                    lossmap_write[i][k].append(v.detach().cpu())

        losmap_sum = torch.tensor([sum(zip(*filter(lambda x: x[0] != 'ious' and x[1], lossmap.items()))[1]) for lossmap in lossmapes])
        checklosses.append(losmap_sum.detach().cpu())

    checkloss_mean = torch.stack(checklosses).mean(0)
    if writer:
        writer.add_scalar('%s_%s_losses/losses' % (name, pretitle), checkloss_mean.sum(), global_step=ep)
        for i, lossmap in enumerate(lossmap_write):
            writer.add_scalar('%s_%s_losses/%d' % (name, pretitle, conf.grids[i]), checkloss_mean[i], global_step=ep)
            if optimizer:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                writer.add_scalar('%s_%s_losses/eta' % (name, pretitle), lr, global_step=ep)

            for k, v in lossmap.iteritems():
                nv = filter(lambda x: x >= 0, v)
                if nv:
                    writer.add_scalar('%s_%s_%s/%d' % (name, pretitle, k, conf.grids[i]), torch.stack(nv).mean(), global_step=ep)

    return float(checkloss_mean.sum())

def main(epoch=2, eta=0.001, lamb=1e-4, gmomentum=0.9, bat=32, method_optim='sgd', models=None, checkpoint=None, cuda='cuda', codename=None, jitterimg=True, shift=True, maxsize=0, summary=True):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    if not models:
        models = util.extact_model(checkpoint)

    if models == 'darknet53':
        net = darknet53(conf.class_num)
    else:
        net = s_resnet(models, True, conf.class_num)

    net = net.to(device)

    SGD = method_optim.lower() == 'sgd'
    if SGD:
        optimizer = torch.optim.SGD(net.parameters(), lr=eta, momentum=gmomentum, weight_decay=lamb)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=eta, weight_decay=lamb)

    print 'device:', device, torch.cuda.get_device_name(device) if torch.cuda.is_available() and cuda != 'cpu' else 'cpu'
    print 'jitterimg', jitterimg
    print 'shift', shift
    print 'optim:', 'SGD' if SGD else 'ADAM'

    if checkpoint:
        print 'loading checkpoint', checkpoint
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    print 'class num', conf.class_num
    print 'init trainset...'
    trainset = loader.imgloader(conf.trainroot, conf.input_size, jitterimg=jitterimg, shift=shift, maxsize=maxsize, cache_ori=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bat, shuffle=True, collate_fn=loader.collate, num_workers=1)
    print 'train_size:%s' % len(trainset)

    print 'init validset...'
    validset = loader.imgloader(conf.validroot, conf.input_size, jitterimg=False, shift=False, maxsize=maxsize, cache_ori=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=bat, shuffle=True, collate_fn=loader.collate, num_workers=1)
    print 'valid_size:%s' % len(validset)

    pretitle = models
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
            if not np.isnan(checkloss) and checkloss < bestloss:
                bestloss = checkloss
                save_weight(net, codename, pretitle, bestloss)

        lr_optim.step()

        desc = 'ETA:%.5f tL:%.1f cL:%.1f bL:%.1f' % (optimizer.param_groups[0]['lr'], trainloss, checkloss, bestloss)
        Trange.set_description_str(desc)

    save_weight(net, '%s.end%s' % (codename, ep), pretitle, checkloss, True)

if __name__ == "__main__":
    fire.Fire(main)
