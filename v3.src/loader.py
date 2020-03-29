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
from base import init_classes20, init_classes80
from base import libshift
from base import libiou
import conf

def loadimg_part(sop, classmap):
    recs = []
    objs = []

    jpgname = sop.filename.text
    if int(sop.size.depth.text) != 3:
        return None

    for _, it in enumerate(sop.findAll("object")):
        xmin, ymin = float(it.bndbox.xmin.text), float(it.bndbox.ymin.text)
        xmax, ymax = float(it.bndbox.xmax.text), float(it.bndbox.ymax.text)
        rec = torch.tensor((xmin, ymin, xmax, ymax))
        recs.append(rec)

        label = it.find("name").text
        objs.append(classmap[label])

    return jpgname, torch.stack(recs), torch.tensor(objs)

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

    idmap = dict((y, x) for x, y in init_classes80.classmap_coco.items())
    key_bbox = 'bbox'
    key_label = 'label'

    for _, it in enumerate(dt['annotations']):
        labelname = idmap[it['category_id']]
        if labelname not in init_classes80.classmap:
            continue

        labelid = init_classes80.classmap[labelname]
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

def rec2cent(rec, (real_x, real_y), best_grids):
    '''
    >>> rec2cent(torch.tensor([[50, 50, 80, 280], [100, 100, 200, 200]]), (640, 480), 7)
    (tensor([[0.7109, 0.4062, 0.0469, 0.4792],
            [0.6406, 0.1875, 0.1562, 0.2083]]), tensor([[0, 2],
            [1, 2]], dtype=torch.int32))
    '''
    rec = rec.float()
    real_xy = torch.tensor([real_x, real_y]).float()
    r = torch.zeros(rec.shape)

    unitsize = torch.stack([real_xy / grid for grid in best_grids])

    cxy = (rec[:, :2] + rec[:, 2:]) * 0.5 / unitsize
    r[:, :2], cellID = np.modf(cxy.detach().cpu())

    wh = (rec[:, 2:] - rec[:, :2]) / real_xy
    r[:, 2:] = wh.clamp(1e-4)
    return r, cellID.int()

class imgloader(data.Dataset):
    def __init__(self, data_roots, isize, jitterimg, shift, maxsize, cache_ori):
        self.jitter = transforms.ColorJitter(0.35, 0.35, 0.35, 0.1) if jitterimg else None
        self.hflip = transforms.RandomHorizontalFlip(1.0) if jitterimg else None
        self.shift = shift
        self.tot = transforms.ToTensor()
        self.toi = transforms.ToPILImage()
        self.normimg = transforms.Normalize(mean=[0.4609, 0.4387, 0.4032], std=[0.2429, 0.2377, 0.2413])

        self.cache_ori = cache_ori
        self.isize = isize

        self.imgnames = []
        self.hismap = {}

        def addhis(objs):
            for _obj in objs:
                obj = int(_obj)
                if obj not in self.hismap:
                    self.hismap[obj] = 0
                self.hismap[obj] += 1

        for data_root in data_roots:
            if 'voc' in data_root.lower():
                rdir = '%s/Annotations' % data_root
                listdir = os.listdir(rdir)
                for _, name in enumerate(sorted(listdir, key=lambda x: hash(x))):
                    if maxsize and len(self.imgnames) >= maxsize:
                        break

                    xml_path = '%s/%s' % (rdir, name)
                    r = loadimg(xml_path, init_classes20.classmap)
                    if not r:
                        continue

                    jpgname, recs, objs = r
                    imgid = jpgname.split('.')[0]
                    self.imgnames.append(['%s/JPEGImages/%s' % (data_root, jpgname), recs, objs, imgid])
                    addhis(objs)
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
                    addhis(objs)

    def __getitem__(self, index):
        fname, recs, objs, imgid = self.imgnames[index]
        with open('%s' % (fname)) as fd:
            img = Image.open(fd)
            if img.mode.lower() != 'rgb':
                img = img.convert('RGB')

            if self.hflip and random.random() > 0.5:
                img = self.hflip(img)
                r0 = img.width - recs[:, 2]
                r2 = img.width - recs[:, 0]
                recs[:, 0] = r0
                recs[:, 2] = r2

            if self.jitter:
                img = self.jitter(img)

            if self.shift and random.random() > 0.5:
                imgx = self.tot(img)
                imgx, recs, objs = libshift.randomCrop(imgx, recs, objs)
                img = self.toi(imgx)

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
            if conf.best_anchor_threshold:
                def _mk_indx(xs):
                    r = torch.arange(xs.nelement())[xs > conf.best_anchor_threshold]
                    if len(r) == 0:
                        r = xs.max(dim=-1)[1].unsqueeze(0)
                    return r

                anchorinds = [_mk_indx(iou_anchor) for iou_anchor in iou_anchors]

                recs = torch.cat([recs[i].repeat(len(x), 1) for i, x in enumerate(anchorinds)])
                objs = torch.cat([objs[i].repeat(len(x), 1) for i, x in enumerate(anchorinds)])
                anchorinds = torch.cat(anchorinds)
            else:
                anchorinds = [iou_anchor.max(dim=-1)[1] for iou_anchor in iou_anchors]

            best_rsind = [x / len(conf.grids) for x in anchorinds]
            best_grid = [conf.grids[ind] for ind in best_rsind]

            best_anchorind = [x % len(conf.grids) for x in anchorinds]

            cent = rec2cent(recs, img.size, best_grid)
            cinfo = zip(cent[0], cent[1], best_rsind, best_anchorind)

            img = img.resize((self.isize, self.isize))
            imgx = self.tot(img)
            imgx = self.normimg(imgx)

            img.close()

        return imgx_ori, imgx, cinfo, objs, imgid

    def __len__(self):
        return len(self.imgnames)

def collate(item):
    imgx_ori, imgx, cinfo, objs, imgid = zip(*item)
    return imgx_ori, torch.stack(imgx), cinfo, objs, imgid

def main():
    def showimg(title, imgx, cinfo, ps, objs):
        h, w = imgx.shape[1:]
        box_nms = []
        for i, (points, cellid, best_rsind, _) in enumerate(cinfo):
            grid = conf.grids[best_rsind]
            rec = libiou.cent2rec((w, h), grid, cellid, points)
            cind = objs[i]
            if title == 'ori':
                print i, cind, 'rec', rec
            box_nms.append((ps[i], rec, None, cind, 0.99, 0, 0))

        util.showimg(title, imgx, box_nms, [])

    root = ['vocdata/voc2007_train']
    maxsize = 4
    resize = 416
    trainset = imgloader(root, resize, jitterimg=True, shift=True, maxsize=maxsize, cache_ori=True)
    trainset.normimg = transforms.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])

    print 'trainset size', len(trainset)
    for k, v in sorted(trainset.hismap.items(), key=lambda (x, y): y):
        print ' ', k, init_classes20.idmap[k], v

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate, num_workers=0)
    for it, (imgx_oris, imgxs, cent, objs, imgid) in enumerate(trainloader):
        assert len(imgxs) == len(objs)
        for i, imgx in enumerate(imgx_oris):
            assert len(cent[i]) == len(objs[i])
            ps = torch.ones(len(cent[i]))
            for title, img in [('ori', imgx), ('resize', imgxs[i])]:
                showimg('%s_%s' % (title, imgid[i]), img, cent[i], ps, objs[i])

if __name__ == "__main__":
    from base import util
    main()
