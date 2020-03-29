import sys
import os
import random
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import json
import init_classes
import libshift
import libiou
from loader import imgloader
import conf

sys.path.append("../")
sys.path.append("../../")

'''
vocdata/voc2012 size 17125
  bus(5) 685
  train(18) 704
  cow(9) 771
  diningtable(10) 800
  motorbike(13) 801
  horse(12) 803
  bicycle(1) 837
  sofa(17) 841
  tvmonitor(19) 893
  aeroplane(0) 1002
  boat(3) 1059
  sheep(16) 1084
  pottedplant(15) 1202
  bird(2) 1271
  cat(7) 1277
  bottle(4) 1561
  dog(11) 1598
  car(6) 2492
  chair(8) 3056
  person(14) 17401

vocdata/voc2007 size 5011
  bus(5) 272
  diningtable(10) 310
  train(18) 328
  aeroplane(0) 331
  sheep(16) 353
  cow(9) 356
  tvmonitor(19) 367
  cat(7) 389
  motorbike(13) 390
  boat(3) 398
  horse(12) 406
  bicycle(1) 418
  sofa(17) 425
  dog(11) 538
  bird(2) 599
  pottedplant(15) 625
  bottle(4) 634
  chair(8) 1432
  car(6) 1644
  person(14) 5447

vocdata/voc2007_test size 4952
  bus(5) 254
  diningtable(10) 299
  train(18) 302
  aeroplane(0) 311
  sheep(16) 311
  cow(9) 329
  tvmonitor(19) 361
  motorbike(13) 369
  cat(7) 370
  bicycle(1) 389
  boat(3) 393
  horse(12) 395
  sofa(17) 396
  dog(11) 530
  bird(2) 576
  pottedplant(15) 592
  bottle(4) 657
  chair(8) 1374
  car(6) 1541
  person(14) 5227
'''

def main():
    def showimg(title, imgx, cents, grid, ps, objs):
        h, w = imgx.shape[1:]
        box_nms = []
        points, cellid = cents
        for i, point in enumerate(points):
            rec = libiou.cent2rec((w, h), grid, cellid[i], point)
            cind = objs[i]
            if title == 'ori':
                print i, cind, 'rec', rec
            box_nms.append((ps[i], rec, None, cind, 0.99))

        util.showimg(title, imgx, box_nms, [])

    maxsize = 0
    resize = 640
    grid = resize / 32
    for root in ['coco/val2017', 'coco/train2017']:
        trainset = imgloader([root], resize, grid=grid, jitterimg=True, shift=True, maxsize=maxsize, cache_ori=True)

        print '%s size' % root, len(trainset)
        for k, v in sorted(trainset.hismap.items(), key=lambda (x, y): y):
            print '  %s(%s)' % (init_classes.idmap[k], k), v
        print

    trainset = imgloader(conf.trainroot, resize, grid=grid, jitterimg=True, shift=True, maxsize=maxsize, cache_ori=True)
    print 'all size', len(trainset)
    for k, v in sorted(trainset.hismap.items(), key=lambda (x, y): y):
        print '  %s(%s)' % (init_classes.idmap[k], k), v
    print

if __name__ == "__main__":
    import util
    main()
