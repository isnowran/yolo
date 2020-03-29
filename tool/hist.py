if __name__ == "__main__":
    import sys
    sys.path.append('../')

import os
import random
import xml.etree.ElementTree as ET
import json
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import init_classes
from base import libshift
import libiou
import conf
from kmeans import kmeans, avg_iou

def loadimg(image_xml_path, classmap):
    annot = ET.parse(image_xml_path)

    recs = []
    objs = []

    jpgname = annot.find('filename').text.lower().strip()

    for obj in annot.findall('object'):
        xmin, xmax, ymin, ymax = [float(obj.find('bndbox').find(tag).text) for tag in ["xmin", "xmax", "ymin", "ymax"]]
        label = obj.find('name').text
        rec = torch.tensor((xmin, ymin, xmax, ymax))
        recs.append(rec)

        objs.append(classmap[label])

    return jpgname, torch.stack(recs), torch.tensor(objs)

def loadimg_coco(jsonname):
    with open(jsonname) as fd:
        dt = json.load(fd)

    imgs = {}
    objs = {}

    ekeys = 'file_name', 'height', 'width'
    for it in dt['images']:
        ID = it['id']
        objs[ID] = {}
        for ekey in ekeys:
            objs[ID][ekey] = it[ekey]

    idmap = dict((y, x) for x, y in init_classes.classmap_coco.items())
    key_bbox = 'bbox'
    key_label = 'label'

    for _, it in enumerate(dt['annotations']):
        labelname = idmap[it['category_id']]
        if labelname not in init_classes.classmap:
            continue

        labelid = init_classes.classmap[labelname]
        key = it['image_id']
        if key not in imgs:
            imgs[key] = {key_bbox:[], key_label:[]}
            imgs[key].update(objs[key])

        box = torch.tensor(it[key_bbox]).float()
        box[2:] = box[2:].ceil() + box[:2]
        imgs[key][key_bbox].append(box.floor())
        imgs[key][key_label].append(labelid)

    for k in imgs.keys():
        if len(imgs[k][key_label]) > conf.coco_maxobj:
            imgs.pop(k)
            continue

        for ID, R in conf.seedmap.items():
            size = 1.0 * imgs[k][key_label].count(ID)
            r = size / len(imgs[k][key_label])
            if r > R:
                imgs.pop(k)
                break

    return imgs

class imgloader(data.Dataset):
    def __init__(self, data_roots, isize, jitterimg, shift, maxsize, cache_ori):
        self.jitter = transforms.ColorJitter(0.35, 0.35, 0.35, 0.1) if jitterimg else None
        self.hflip = transforms.RandomHorizontalFlip(1.0) if jitterimg else None
        self.shift = shift
        self.tot = transforms.ToTensor()
        self.toi = transforms.ToPILImage()
        self.normimg = transforms.Normalize(mean=[0.47, 0.45, 0.39], std=[0.23, 0.23, 0.23])

        self.cache_ori = cache_ori
        self.isize = isize

        self.imgnames = []
        imgloader.hismap = {}
        imgloader.hismap2 = {}

        def addhis(objs, hismap):
            for _obj in objs:
                obj = int(_obj)
                if obj not in hismap:
                    hismap[obj] = 0
                hismap[obj] += 1

        for data_root in data_roots:
            if 'voc' in data_root.lower():
                rdir = '%s/Annotations' % data_root
                listdir = os.listdir(rdir)
                for _, name in enumerate(sorted(listdir, key=lambda x: hash(x))):
                    if maxsize and len(self.imgnames) >= maxsize:
                        break

                    xml_path = '%s/%s' % (rdir, name)
                    r = loadimg(xml_path, init_classes.classmap)
                    if not r:
                        continue

                    jpgname, recs, objs = r
                    imgid = jpgname.split('.')[0]
                    self.imgnames.append(['%s/JPEGImages/%s' % (data_root, jpgname), recs, objs, imgid])
                    #addhis(objs)
            else:
                root, name = os.path.split(data_root)
                jsonfile = '%s/annotations/instances_%s.json' % (root, name)
                imgmaps = loadimg_coco(jsonfile)
                for _, imgmap in sorted(imgmaps.iteritems()):
                    if maxsize and len(self.imgnames) >= maxsize:
                        break

                    fname = imgmap['file_name']
                    recs = torch.stack(imgmap['bbox'])
                    objs = torch.tensor(imgmap['label'])
                    self.imgnames.append(['%s/%s/%s' % (root, name, fname), recs, objs, fname])
                    #addhis(objs)

    def __getitem__(self, index):
        def addhis(objs, hismap):
            for _obj in objs:
                obj = int(_obj)
                if obj not in hismap:
                    hismap[obj] = 0
                hismap[obj] += 1

        fname, recs, objs, imgid = self.imgnames[index]
        with open('%s' % (fname)) as fd:
            img = Image.open(fd)
            if img.mode.lower() != 'rgb':
                img = img.convert('RGB')

            imgx_ori = self.tot(img) if self.cache_ori else None

            recs416 = recs.clone()
            recs416[:, 0:3:2] *= (1.0 * self.isize / img.width)
            recs416[:, 1:4:2] *= (1.0 * self.isize / img.height)

            # iou anchor
            copy_anchors = torch.zeros(len(conf.anchors), 4)
            copy_anchors[:, 2:] = conf.anchors

            copy_whs = torch.zeros_like(recs416)
            copy_whs[:, 2:] = recs416[:, 2:] - recs416[:, :2]

            iou_anchors = [libiou.cpu_iou_allbyall(copy_anchors, copy_wh)[0] for copy_wh in copy_whs]
            anchorinds = [iou_anchor.max(dim=-1)[1] for iou_anchor in iou_anchors]

            best_rsind = [x / len(conf.grids) for x in anchorinds]
            best_grid = [conf.grids[ind] for ind in best_rsind]
            best_anchorind = [x % len(conf.grids) for x in anchorinds]
            img.close()

            addhis(best_grid, imgloader.hismap)
            addhis(anchorinds, self.hismap2)

        return recs[:, 2:] - recs[:, :2], recs416[:, 2:] - recs416[:, :2]

    def __len__(self):
        return len(self.imgnames)

def main(resize, method):
    root = ['coco/val2017', 'vocdata/voc2007_train']
    root = ['vocdata/voc2007_train', 'vocdata/voc2012']
    maxsize = 40000
    trainset = imgloader(root, resize, jitterimg=False, shift=False, maxsize=maxsize, cache_ori=False)

    s = time.time()
    print 'trainset size', len(trainset), 'class_num', conf.class_num
    for k, v in sorted(trainset.hismap.items(), key=lambda (x, y): y):
        print ' ', k, init_classes.idmap[k], v

    wh_ori, wh_416 = [], []
    for recs, recs416 in trainset:
        wh_ori.extend(recs)
        wh_416.extend(recs416)

    if method == 'hist':
        print 'his1'
        for k, v in sorted(imgloader.hismap.items(), key=lambda (x, y): y):
            print ' ', k, v

        print 'his2'
        for k, v in sorted(imgloader.hismap2.items(), key=lambda (x, y): y):
            print ' ', k, v
    else:
        data = torch.stack(wh_416).numpy()
        out = kmeans(data, 9)
        sout = torch.tensor((sorted(out, key=lambda(x, y): x*y, reverse=True))).int()
        print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
        print("Boxes")
        ns = [13, 26, 52]
        for i, it in enumerate(sout.view(3, -1, 2)):
            print '_anchor%s = %s' % (ns[i], repr(it))

    print time.time() - s

if __name__ == "__main__":
    import time
    main(int(sys.argv[1]), sys.argv[2])
