from base.init_classes_base import classmap_voc as classmap_voc

classmap = dict(zip(classmap_voc, range(len(classmap_voc))))
idmap = dict((y, x) for x, y in classmap.items())

classmap['airplane'] = classmap['aeroplane']
classmap['dining table'] = classmap['diningtable']
classmap['motorcycle'] = classmap['motorbike']
classmap['potted plant'] = classmap['pottedplant']
classmap['couch'] = classmap['sofa']
classmap['tv'] = classmap['tvmonitor']

if __name__ == "__main__":
    for i, (k, v) in enumerate(classmap.items()):
        print i, k, v

    print 'class_num', max(classmap.values()) + 1
    for c in class_voc:
        assert c in classmap
