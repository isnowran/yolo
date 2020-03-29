#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:35:22 2019

@author: binary
"""

from PIL import ImageDraw
import torchvision.transforms as transforms
import torch

from libiou import cpu_iou
import conf

def check_grad(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    pmax = []
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        pmax.append(p.max().detach().cpu())
    return torch.tensor(total_norm ** (1. / norm_type)), torch.tensor(pmax)


def showimg(title, imgx, box_nms, bboxs=[]):
    img = transforms.ToPILImage()(imgx)
    draw = ImageDraw.Draw(img)

    c = 'black'
    for _, rec in enumerate(bboxs):
        xmin, ymin, xmax, ymax = rec
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        draw.rectangle(rec.numpy(), outline=c)
        draw.ellipse((cx, cy, cx+5, cy+5), fill=c)

    for p, rec, c, cind in box_nms:
        #draw.text((rec[0]+3, rec[1]+2), '%.1f-%s:%.1f %d %s' % (p, init_classes.idmap[int(cind)], cval, grid, anchor))
        draw.text((rec[0]+3, rec[1]+2), '%.1f-%s:%.1f' % (p, conf.init_classes.idmap[int(cind)], c))
        for i in (-1, 0, 1):
            draw.rectangle((rec+i).tolist(), outline=conf.colormap[cind])

    draw.text((5, 5), '%s' % title, fill='black')
    with open('img_out/%s.jpg' % title, 'w+') as fd:
        img.save(fd)
    img.close()


def nms_gpu(p, pbox, c, cind, T_iou):
    x1 = pbox[:, 0]
    y1 = pbox[:, 1]
    x2 = pbox[:, 2]
    y2 = pbox[:, 3]

    scores = p * c
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=1e-12)
        h = torch.clamp(yy2 - yy1, min=1e-12)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr < T_iou)[0]
        order = order[inds + 1]

    if not keep:
        return []

    keep = torch.stack(keep)
    return zip(p[keep], pbox[keep], c[keep], cind[keep])


def nms_cpu(p, pbox, c, cind, T_iou):
    droped = []
    picked = []
    boxes = sorted(zip(p, pbox, c, cind), key=lambda x: x[0]*x[2], reverse=True)
    for i, (_, rec1, _, _) in enumerate(boxes):
        if i in droped:
            continue

        picked.append(i)

        for j, (_, rec2, _, _) in enumerate(boxes):
            if j <= i:
                continue

            iou = cpu_iou(rec1, rec2)[0]
            if iou > T_iou:
                droped.append(j)

    return [x for i, x in enumerate(boxes) if i in picked]


def nms(p, pbox, c, cind, T_p, T_iou, ver='gpu'):
    try:
        p = torch.cat(p, 0)
        pbox = torch.cat(pbox, 0)
        c = torch.cat(c, 0)
        cind = torch.cat(cind, 0)
    except:
        # ver == 'v2'
        pass

    mask = (p*c) > T_p
    fun = nms_gpu if ver == 'gpu' else nms_cpu
    return fun(p[mask], pbox[mask], c[mask], cind[mask], T_iou)


def cent2rec_netout((w, h), grid, pbox):
    y, x, an_size = pbox.shape[:3]

    # torch.linspace(0, x-1, x).repeat(y, 1).repeat(a, 1).t().view(y, x, a).transpose(0, 1)
    gx = torch.linspace(0, x-1, x).repeat(y, 1).to(pbox.device)

    # torch.linspace(0, y-1, y).repeat(x, 1).repeat(a, 1).t().view(y, x, a)
    gy = torch.linspace(0, y-1, y).repeat(x, 1).t().to(pbox.device)

    for i in range(an_size):
        pbox[:, :, i, 0] += gx
        pbox[:, :, i, 1] += gy

    # cent2rec
    r = torch.zeros_like(pbox)
    half_wh = 0.5 * pbox[:, :, :, 2:]
    r[:, :, :, :2] = pbox[:, :, :, :2] - half_wh
    r[:, :, :, 2:] = pbox[:, :, :, :2] + half_wh

    unit_x, unit_y = 1.0 * w / grid, 1.0 * h / grid
    r[:, :, :, 0] *= unit_x
    r[:, :, :, 0].clamp_(min=1.0)

    r[:, :, :, 1] *= unit_y
    r[:, :, :, 1].clamp_(min=1.0)

    r[:, :, :, 2] *= unit_x
    r[:, :, :, 2].clamp_(max=w-1.0)

    r[:, :, :, 3] *= unit_y
    r[:, :, :, 3].clamp_(max=h-1.0)

    return r


def cell_decode(rs, anchors):
    p = rs[:, :, :, conf.loc_p]
    pbox = rs[:, :, :, conf.loc_box:conf.loc_box+4].clone()
    pbox[:, :, :, 2:] = pbox[:, :, :, 2:].exp() * anchors
    classes = rs[:, :, :, conf.loc_cls:conf.loc_cls+conf.class_num].softmax(-1)
    return p, pbox, classes


def create_box(rs, (w, h), grid, anchor):
    p, pbox, classes = cell_decode(rs, anchor)
    rec = cent2rec_netout((w, h), grid, pbox)
    maxv, maxind = classes.max(dim=-1)
    return p, rec, maxv, maxind


def extact_model(checkpoint):
    """
    >>> extact_model('endkeyresnet152.v1')
    'resnet152.v1'

    >>> extact_model('endkey.resnet50.v3')
    'resnet50.v3'

    >>> extact_model('endkey.resnet152')

    """

    if not checkpoint:
        return checkpoint

    keys = ['darknet53', 'darknet19']
    key1 = '50', '101', '152'

    for i in key1:
        keys.append("resnet%s" % i)

    for it in keys:
        if it in checkpoint:
            return it

    return None
