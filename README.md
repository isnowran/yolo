## yolo_v2 and vocdata, video detect
> python cvideo.py --mp4=video/379489.mp4 --T_p=0.35 --checkpoint=checkpoint/v2.resnet152.dump
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/v2_voc_411907.gif)

## yolo_v2 and vocdata, image detect
> python valid.py --checkpoint=checkpoint/v2.resnet152.dump --maxsize=100 --drawimg=True --debug=False --models=resnet152 --T_p=0.35

![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/24_007339.jpg)
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/28_007237.jpg)
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/31_000467.jpg)

## train v2/v3, voc/coco, resnet/darknet
> python main.py --cuda=cuda:1 --epoch=180 --eta=0.0001 --bat=16 --codename=voc --models=darknet53 --summary=True --jitterimg=True --shift=True

## view loss
> tensorboard --logdir runs --bind_all
