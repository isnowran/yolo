#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:20:52 2019

@author: binary
"""
import sys
sys.path.append('..')

import os
import torch
import cv2
import fire
import time

import conf
from base.snet import s_resnet
from base.dnet53 import darknet53_v3 as darknet53
from base import libnet, util

def getfs(video, isize):
    success, frame = video.read()
    if not success:
        return None, frame, None

    imgx = torch.tensor(cv2.resize(frame, (isize, isize)), dtype=torch.float)
    mean = torch.tensor([0.47, 0.45, 0.39])
    std = torch.tensor([0.23, 0.23, 0.23])
    imgx = (imgx / 255.0 - mean) / std
    imgx = imgx.permute(2, 0, 1)
    imgx = imgx.unsqueeze(0)
    return imgx, frame, success

def proc(net, T_p, T_iou, mp4, device):
    isize = conf.input_size
    video = cv2.VideoCapture(mp4)
    fps = video.get(cv2.CAP_PROP_FPS)
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    sizex, sizey = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    xrate, yrate = 1.0 * sizex / isize, 1.0 * sizey / isize

    p, f = os.path.split(mp4)
    f2 = 'c_%s' % f
    mp5 = os.path.join(p, f2)
    videoWriter = cv2.VideoWriter(mp5, cv2.VideoWriter_fourcc(*'MP4V'), fps, (sizex, sizey))

    c = 0
    s = time.time()
    while video.isOpened():
        c += 1
        imgx, frame, success = getfs(video, isize)
        if not success:
            break

        if c and c % int(fps) == 0:
            print '- frame rate %d, sec' % fps, c / fps, 'fps', c / (time.time() - s)

        imgx_t = imgx.to(device)
        with torch.no_grad():
            rss = [libnet.refine(r, conf.class_num+5) for r in net(imgx_t)]

        # ap
        box_all = []
        ps = []
        recs = []
        cs = []
        cinds = []
        for rsind, rs in enumerate(rss):
            rs = rs.squeeze()
            p, rec, maxv, maxind = util.create_box(rs, (sizex, sizey), conf.grids[rsind], conf.anchor_split[rsind].to(device))

            ps.append(p.reshape(-1))
            recs.append(rec.view(-1, rec.shape[-1]))
            cs.append(maxv.reshape(-1))
            cinds.append(maxind.reshape(-1))

        box_pure = util.nms(ps, recs, cs, cinds, T_p, T_iou)

        for p, bbox, cval, cind in box_pure:
            cname = conf.init_classes.classmap_voc[cind]
            color = conf.colormap[cind]

            box = tuple(bbox.int().tolist())
            cv2.putText(frame, '%s - %.2f:%.2f' % (cname, p, cval), box[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            cv2.rectangle(frame, box[:2], box[2:], color, 1)
            print c, 'fps', '%.2f' % (c / (time.time() - s)), \
                     '%.2f x %.2f = %.2f' % (p.detach(), cval.detach(), p.detach() * cval.detach()), 'boxes', box, cname
        videoWriter.write(frame)

    video.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    print 'convert to', mp5
    print 'fps', fps, 'frameCount', frameCount, 'size', (sizex, sizey)
    print 'isize', isize, 'ratex, ratey', xrate, yrate

def main(mp4, T_p=0.75, T_iou=0.5, checkpoint=None, cuda='cpu', models=None):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    if not models:
        models = util.extact_model(checkpoint)

    if models == 'darknet53':
        net = darknet53(conf.class_num)
    else:
        net = s_resnet(models, False, conf.class_num)

    net = net.to(device)
    net.eval()

    if checkpoint:
        print 'load checkpoint', checkpoint
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    with torch.no_grad():
        proc(net, T_p, T_iou, mp4, device)

if __name__ == "__main__":
    fire.Fire(main)
