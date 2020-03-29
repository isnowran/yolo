#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:32:59 2019

@author: binary
"""

import random
import torch
import cv2

def randomCrop(imgx, boxes, objs, rr=0.75):
    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    _, height, width = imgx.shape
    h = random.uniform(rr * height, height)
    w = random.uniform(rr * width, width)
    x = random.uniform(0, width - w)
    y = random.uniform(0, height - h)
    x, y, h, w = int(x), int(y), int(h), int(w)

    center = center - torch.FloatTensor([[x, y]]).expand_as(center)
    mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
    mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
    mask = (mask1 & mask2).view(-1, 1)

    boxes_in = boxes[mask.expand_as(boxes)].view(-1, boxes.shape[-1])
    objs_in = objs[mask.squeeze()]
    if len(boxes_in) == 0:
        return imgx, boxes, objs

    box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)
    boxes_in = boxes_in - box_shift

    boxes_in[:, 0].clamp_(min=0, max=w-1)
    boxes_in[:, 1].clamp_(min=0, max=h-1)
    boxes_in[:, 2].clamp_(min=1, max=w)
    boxes_in[:, 3].clamp_(min=1, max=h)

    img_croped = imgx[:, y:y+h, x:x+w]
    check = boxes_in[:, [2, 3]] > boxes_in[:, [0, 1]]
    mask = check.all(dim=1)

    if not mask.any():
        return imgx, boxes, objs

    return img_croped, boxes_in[mask], objs_in[mask]


def randomShift(imgx, boxes, objs, rr=0.2):
    c, height, width = imgx.shape
    after_shfit_image = torch.zeros((c, height, width), dtype=imgx.dtype)
    after_shfit_image[0, :] = 104.0 / 255
    after_shfit_image[1, :] = 117.0 / 255
    after_shfit_image[2, :] = 123.0 / 255

    shift_x = random.uniform(-width * rr, width * rr)
    shift_y = random.uniform(-height * rr, height * rr)

    if shift_x >= 0 and shift_y >= 0:
        after_shfit_image[:, int(shift_y):, int(shift_x):] = imgx[:, :height-int(shift_y), :width-int(shift_x)]
    elif shift_x >= 0 and shift_y < 0:
        after_shfit_image[:, :height+int(shift_y), int(shift_x):] = imgx[:, -int(shift_y):, :width-int(shift_x)]
    elif shift_x < 0 and shift_y >= 0:
        after_shfit_image[:, int(shift_y):, :width+int(shift_x)] = imgx[:, :height-int(shift_y), -int(shift_x):]
    elif shift_x < 0 and shift_y < 0:
        after_shfit_image[:, :height+int(shift_y), :width+int(shift_x)] = imgx[:, -int(shift_y):, -int(shift_x):]

    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
    center = center + shift_xy
    mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
    mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
    mask = (mask1 & mask2).view(-1, 1)

    boxes_in = boxes[mask.expand_as(boxes)].view(-1, boxes.shape[-1])
    objs_in = objs[mask.expand_as(objs)].view(-1, objs.shape[-1])
    if len(boxes_in) == 0:
        return imgx, boxes, objs

    box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
    boxes_in = boxes_in + box_shift

    j = boxes_in[:, [0, 2]] < 0
    j += boxes_in[:, [0, 2]] > width
    j += boxes_in[:, [1, 3]] < 0
    j += boxes_in[:, [1, 3]] > height

    mask = j.sum(dim=1) == 0
    boxes_in = boxes_in[mask]
    if len(boxes_in) == 0:
        return imgx, boxes, objs

    return after_shfit_image, boxes_in, objs_in[mask]


def randomScale(imgx, boxes, rr=0.2):
    scale = random.uniform(1-rr, 1+rr)
    _, height, width = imgx.shape
    imgx2 = imgx.permute((1, 2, 0)).numpy()
    imgx3 = cv2.resize(imgx2, (int(width*scale), height))
    scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
    boxes = boxes * scale_tensor
    return torch.tensor(imgx3).permute(2, 0, 1), boxes


def draw2show(img, boxes, c):
    draw = ImageDraw.Draw(img)
    for bb in boxes:
        draw.rectangle(bb.numpy(), outline=c)
    img.show()


def main():
    #test shift
    tt = transforms.ToTensor()
    ti = transforms.ToPILImage()

    box = [torch.tensor((100.0, 100.0, 350.0, 350.0)), torch.tensor((50.0, 150.0, 200.0, 250.0))]
    img = Image.open("/Users/binary/Downloads/rose_dog.jpg")
    boxes = torch.stack(box)
    draw2show(img, boxes, 'red')
    print 'size 1', img.size

    imgx, boxes = randomScale(tt(img), boxes)
    img2 = ti(imgx)
    draw2show(img2, boxes, 'green')
    print 'size 2', img2.size, imgx.shape

    imgx, boxes, _ = randomCrop(imgx, boxes, boxes.clone())
    img3 = ti(imgx)
    draw2show(img3, boxes, 'blue')
    print 'size 3', img3.size, imgx.shape

if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    from PIL import ImageDraw

    main()
