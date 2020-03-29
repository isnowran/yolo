from base.init_classes_base import classmap_voc as classmap_voc

classmap = dict(zip(classmap_voc, range(len(classmap_voc))))
idmap = dict((y, x) for x, y in classmap.items())

if __name__ == "__main__":
    for i, (k, v) in enumerate(classmap.items()):
        print(i, k, v)

    print('class_num', max(classmap.values()) + 1)
    for c in classmap_voc:
        assert c in classmap
