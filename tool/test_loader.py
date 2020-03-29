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

sys.path.append("../")
sys.path.append("../../")

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

    key_bbox = 'bbox'
    key_label = 'label'
    for i, it in enumerate(data['annotations']):
        key = it['image_id']
        if key not in imgs:
            imgs[key] = {key_bbox:[], key_label:[]}
            imgs[key].update(objs[key])

        box = torch.tensor(it[key_bbox]).float()
        box[2:] += box[:2]
        imgs[key][key_bbox].append(box)
        imgs[key][key_label].append(it['category_id'] - 1)

    return imgs

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

    def __getitem__(self, index):
        fname, recs, objs, imgid = self.imgnames[index]
        with open('%s' % (fname)) as fd:
            img = Image.open(fd)
            if img.mode.lower() != 'rgb':
                img = img.convert('RGB')

            '''
            if self.hflip and random.random() > 1.:
                img = self.hflip(img)
                r0 = img.width - recs[:, 2]
                r2 = img.width - recs[:, 0]
                recs[:, 0] = r0
                recs[:, 2] = r2

            if self.jitter:
                img = self.jitter(img)
            '''

            if self.shift and random.random() > 0.:
                imgx = self.tot(img)
                imgx, recs, objs = libshift.randomCrop(imgx, recs, objs)
                img = self.toi(imgx)

            imgx_ori = self.tot(img) if self.cache_ori else None
            cents = libiou.rec2cent(recs, img.size, self.grid)

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
    import time
    root = ['coco/val2017']
    maxsize = 0
    resize = 416
    grid = resize / 32
    trainset = imgloader(root, resize, grid=grid, jitterimg=True, shift=True, maxsize=maxsize, cache_ori=True)
    trainset.normimg = transforms.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])

    print 'trainset size', len(trainset)

    for i in range(10):
        print 'i time', i
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, collate_fn=collate, num_workers=0)
        for imgx_oris, imgxs, cents, objs, imgid in trainloader:
            print time.time()
            for cent in cents:
                if not (cent[0][:, 2:] > 0).all():
                    print imgid
                    print cent[0]
                    ee

if __name__ == "__main__":
    import util
    main()
