#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:32:59 2019

@author: binary
"""

import numpy as np
import torch

def cent2rec((w, h), grid, cellID, arg):
    unit_size = torch.tensor((w, h)).float() / grid
    cent_x, cent_y = (cellID.float() + arg[:2]) * unit_size
    width, height = arg[2] * w, arg[3] * h
    minx = cent_x - width * 0.5
    maxx = cent_x + width * 0.5
    miny = cent_y - height * 0.5
    maxy = cent_y + height * 0.5
    return torch.stack([minx, miny, maxx, maxy])


def cent2rec_samecell(cent, grid):
    xy = torch.zeros(cent.shape)
    half_wh = 0.5 * cent[:, 2:] * grid
    xy[:, :2] = cent[:, :2] - half_wh
    xy[:, 2:] = cent[:, :2] + half_wh
    return xy


def cent2rec_anchor(cent):
    xy = torch.zeros(cent.shape)
    half_wh = 0.5 * cent[:, 2:]
    xy[:, :2] = cent[:, :2] - half_wh
    xy[:, 2:] = cent[:, :2] + half_wh
    return xy


def cpu_iou_allbyall(rec_out, rec_dest):
    ''' rec(xmin, ymin, xmax, ymax)'''
    x_LB, y_LB = [x.squeeze(-1) for x in torch.max(rec_out[:, :2], rec_dest[:2]).split(1, dim=-1)]
    x_RT, y_RT = [x.squeeze(-1) for x in torch.min(rec_out[:, 2:4], rec_dest[2:4]).split(1, dim=-1)]

    iou_w, iou_h = x_RT - x_LB, y_RT - y_LB
    iou_area = iou_w * iou_h
    out_area = (rec_out[:, 2] - rec_out[:, 0]) * (rec_out[:, 3] - rec_out[:, 1])
    dest_area = (rec_dest[2] - rec_dest[0]) * (rec_dest[3] - rec_dest[1])
    all_area = out_area + dest_area

    ious = torch.zeros(iou_area.shape)

    ind = (iou_w > 0) & (iou_h > 0)
    ious[ind] = iou_area[ind] / (all_area[ind] - iou_area[ind])
    return ious, (x_LB, y_LB, x_RT, y_RT)


def cpu_iou(rec1, rec2):
    ''' rec(xmin, ymin, xmax, ymax)'''
    x_LB, y_LB = np.max((rec1[0], rec2[0])), np.max((rec1[1], rec2[1]))
    x_RT, y_RT = np.min((rec1[2], rec2[2])), np.min((rec1[3], rec2[3]))

    iou_w, iou_h = x_RT - x_LB, y_RT - y_LB
    iou_area = iou_w * iou_h
    if iou_w < 0 or iou_h < 0:
        iou = torch.tensor(0.0)
    else:
        areas = []
        for retc in (rec1, rec2):
            area = (retc[2] - retc[0]) * (retc[3] - retc[1])
            areas.append(area)

        iou = iou_area / (sum(areas) - iou_area)

    return iou, (x_LB, y_LB, x_RT, y_RT)


def test_iou(fname, rec1, rec2):
    #rec1, rec2 = [0, 1, 3, 3], [1, 2, 3, 2]
    ret, iou_rec = cpu_iou(rec1, rec2)
    print 'iou', ret, iou_rec

    if not fname:
        size = 448
        imgx = torch.zeros((3, size, size))
        img = transforms.ToPILImage()(imgx)
    else:
        img = Image.open(fname)

    draw = ImageDraw.Draw(img)
    for i, rec in enumerate((rec1, rec2, iou_rec)):
        color = 'red'
        if i == 1:
            color = 'blue'
        if i == 2:
            rec = list(rec)
            color = 'yellow'
            xx = 2
            rec[0] += xx
            rec[1] += xx
            rec[2] -= xx
            rec[3] -= xx

        draw.rectangle(rec, outline=color)

    img.show()


def test_cpu_cell():
    isize = 512
    grid = 7
    cell_ID = (3, 3)
    arg = torch.tensor((0.1, 0.2, 0.3, 0.5))
    print cent2rec(isize, (isize, isize), grid, cell_ID, arg)
