#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:53:51 2019

@author: binary
"""

import torch
import fire

import loader
from dnet53 import darknet53
import conf
from collections import OrderedDict as odict
import numpy as np
from main import train

def smain(eta=0.0, lamb=0, gmomentum=0.9, bat=16, method_optim='sgd', cuda='cuda', maxsize=480, checkpoint=None):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    net = darknet53(conf.class_num)
    net = net.to(device)
    if checkpoint:
        print 'loading checkpoint', checkpoint
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    SGD = method_optim.lower() == 'sgd'

    if SGD:
        optimizer = torch.optim.SGD(net.parameters(), lr=eta, momentum=gmomentum, weight_decay=lamb)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=eta, weight_decay=lamb)

    grid = 13
    trainset_2012 = loader.imgloader(['vocdata/voc2012'], conf.input_size, jitterimg=False, shift=False, maxsize=maxsize, cache_ori=False)
    trainset_cocoval = loader.imgloader(['coco/val2017'], conf.input_size, jitterimg=False, shift=False, maxsize=maxsize, cache_ori=False)

    print 'device:', device
    print 'eta:', eta
    print 'optim:', 'SGD' if SGD else 'ADAM'
    print 'train_size', [len(x) for x in [trainset_2012, trainset_cocoval]]

    gradmap = odict()
    glist = odict()

    for trainset in [trainset_cocoval, trainset_2012]:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bat, shuffle=True, collate_fn=loader.collate, num_workers=0)

        for k, v in conf.gmap.items():
            # clean conf.gmap
            for _k in conf.gmap:
                conf.gmap[_k] = 0

            if k not in gradmap:
                print 'init', k
                gradmap[k] = []
                glist[k] = odict()

            conf.gmap[k] = 1.0
            for _, imgx, convtinfo, objs, _ in trainloader:
                train(net, optimizer, imgx, convtinfo, objs, device)

                for name, p in net.named_parameters():
                    if p.grad is None:
                        continue

                    if name not in glist[k]:
                        glist[k][name] = 0
                    glist[k][name] += p.grad.abs().mean().detach().cpu()

    mv = torch.stack(glist[conf.gk_coor].values()).mean()
    for k, vmap in glist.items():
        print k, np.mean(vmap.values()), mv / np.mean(vmap.values())

if __name__ == "__main__":
    '''
    noobj tensor(0.6752) tensor(1.)
    obj tensor(0.0027) tensor(246.4452)
    class tensor(0.0153) tensor(44.0079)
    corr tensor(0.1020) tensor(6.6169)
    '''

    fire.Fire(smain)
