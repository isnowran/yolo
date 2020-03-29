# demo yolo_v2 and vocdata
> cd v2.src

# train voc, resnet or darknet
> python main.py --cuda=cuda:1 --epoch=180 --eta=0.0001 --bat=16 --codename=voc --models=darknet53 --summary=True --jitterimg=True --shift=True

# view loss
> tensorboard --logdir runs --bind_all

# image detect
> python valid.py --checkpoint=checkpoint/v2.resnet152.dump --maxsize=100 --drawimg=True --debug=False --models=resnet152 --T_p=0.35

# video detect
> python cvideo.py --mp4=video/379489.mp4 --T_p=0.35 --checkpoint=checkpoint/v2.resnet152.dump
