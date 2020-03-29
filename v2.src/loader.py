import os
import random
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import json
from base import init_classes80
from base import init_classes20
from base import libshift
import numpy as np
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
        data = json.load(fd)

    imgs = {}
    objs = {}

    ekeys = 'file_name', 'height', 'width'
    for it in data['images']:
        ID = it['id']
        objs[ID] = {}
        for ekey in ekeys:
            objs[ID][ekey] = it[ekey]

    idmap = dict((y, x) for x, y in init_classes80.classmap_coco.items())
    key_bbox = 'bbox'
    key_label = 'label'

    for i, it in enumerate(data['annotations']):
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

def rec2cent(rec, (real_x, real_y), grid):
    '''
    >>> rec2cent(torch.tensor([[50, 50, 80, 280], [100, 100, 200, 200]]), (640, 480), 7)
    (tensor([[0.7109, 0.4062, 0.0469, 0.4792],
            [0.6406, 0.1875, 0.1562, 0.2083]]), tensor([[0, 2],
            [1, 2]], dtype=torch.int32))
    '''
    rec = rec.float()
    real_xy = torch.tensor([real_x, real_y]).float()
    r = torch.zeros(rec.shape)

    unitsize = real_xy / grid

    cxy = (rec[:, :2] + rec[:, 2:]) * 0.5 / unitsize
    r[:, :2], cellID = np.modf(cxy.detach().cpu())

    wh = (rec[:, 2:] - rec[:, :2]) / real_xy
    r[:, 2:] = wh.clamp(1e-4)
    return r, cellID.int()

class imgloader(data.Dataset):
    def __init__(self, data_roots, isize, grid, jitterimg, shift, maxsize, cache_ori):
        self.jitter = transforms.ColorJitter(0.35, 0.35, 0.35, 0.1) if jitterimg else None
        self.hflip = transforms.RandomHorizontalFlip(1.0) if jitterimg else None
        self.shift = shift
        self.tot = transforms.ToTensor()
        self.toi = transforms.ToPILImage()
        self.normimg = transforms.Normalize(mean=[0.47, 0.45, 0.39], std=[0.23, 0.23, 0.23])

        self.cache_ori = cache_ori
        self.isize = isize
        self.grid = grid

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
                for key, imgmap in imgmaps.iteritems():
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
            cents = rec2cent(recs, img.size, self.grid)

            img = img.resize((self.isize, self.isize))
            imgx = self.tot(img)
            imgx = self.normimg(imgx)

            img.close()

        return imgx_ori, imgx, cents, objs, imgid

    def __len__(self):
        return len(self.imgnames)

def collate(item):
    imgx_ori, imgx, cents, objs, imgid = zip(*item)
    return imgx_ori, torch.stack(imgx), cents, objs, imgid

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

    root = ['coco/val2017', 'vocdata/voc2007']
    maxsize = 0
    resize = 640
    grid = resize / 32
    trainset = imgloader(root, resize, grid=grid, jitterimg=True, shift=True, maxsize=maxsize, cache_ori=True)
    trainset.normimg = transforms.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])

    print 'trainset size', len(trainset)
    for k, v in sorted(trainset.hismap.items(), key=lambda (x, y): y):
        print ' ', k, init_classes80.idmap[k], v

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=False, collate_fn=collate, num_workers=0)
    for it, (imgx_oris, imgxs, cents, objs, imgid) in enumerate(trainloader):
        if it > 16:
            break

        assert len(imgxs) == len(cents)
        assert len(imgxs) == len(objs)
        for i, imgx in enumerate(imgx_oris):
            assert len(cents[i][0]) == len(objs[i])
            ps = torch.ones(len(cents[i][0]))
            for title, img in [('ori', imgx), ('resize', imgxs[i])]:
                showimg('%s_%s' % (title, imgid[i]), img, cents[i], grid, ps, objs[i])

if __name__ == "__main__":
    from base import libiou
    from base import util
    main()
