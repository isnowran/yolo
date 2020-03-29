#-*- coding: utf-8 -*-
import os
import sys
import numpy as np

from voc_eval import voc_eval
import init_classes

detpath = sys.argv[1]
detfiles = os.listdir(detpath)

classes = init_classes.classes

aps = []
recs = []
precs = []

vocdata_root = './vocdata'
annopath = '%s/Annotations/' % vocdata_root + '{:s}.xml'
imagesetfile = sys.argv[2]
cachedir = './cache/'

for i, cls in enumerate(classes):
    filename = os.path.join(detpath, cls)
    if not os.path.exists(filename):
        continue

    rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=False)
    aps += [ap]

    print '# proc', cls
    print 'AP for {} = {:.4f}'.format(cls, ap)
    print 'recall for {} = {:.4f}'.format(cls, rec[-1])
    print 'precision for {} = {:.4f}'.format(cls, prec[-1])

print
print '~~~~~~~~'
print 'Mean AP = {:.4f}'.format(np.mean(aps))
print
print '~~~~~~~~'
print 'Results:'
for ap in aps:
    print '{:.3f}'.format(ap)
print '{:.3f}'.format(np.mean(aps))
print '~~~~~~~~'
