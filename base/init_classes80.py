from base.init_classes_base import classmap_coco as classmap_coco
from base.init_classes_base import classmap_voc as classmap_voc

classmap = dict([(k, v-1) for k, v in classmap_coco.items()])
idmap = dict((y, x) for x, y in classmap.items())

# include vocdata
classmap['aeroplane'] = classmap['airplane']
classmap['diningtable'] = classmap['dining table']
classmap['motorbike'] = classmap['motorcycle']
classmap['pottedplant'] = classmap['potted plant']
classmap['sofa'] = classmap['couch']
classmap['tvmonitor'] = classmap['tv']

for c in classmap_voc:
    idmap[classmap[c]] = c

if __name__ == "__main__":
    for k, v in classmap.items():
        print k, v

    print len(classmap), len(idmap)
    print max(classmap.values())
    print 'class_num', max(classmap.values()) + 1
    for c in classmap_voc:
        assert c in classmap
