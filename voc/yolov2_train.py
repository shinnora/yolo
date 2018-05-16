import time
import cv2
import numpy as np
import glob
import os
import renom as rm
from renom.optimizer import Sgd
from renom.utility.trainer import Trainer
from renom.cuda.cuda import set_cuda_active
from renom import cuda
from renom.utility.distributor import NdarrayDistributor
from yolov2 import *
from utils import *

set_cuda_active(True)

# hyper parameters
backup_path = "./weights"
backup_file = "%s/backup.h5" % (backup_path)
pretrained_weight_file = "%s/darknet19_448.h5" % (backup_path)
initial_weight_file = "%s/backup.h5" % (backup_path)
batch_size = 8
epochs = 160
learning_rate = 1e-4
learning_schedules = {
    "0" : 1e-4,
    "10" : 1e-3,
    "60" : 1e-4,
    "90": 1e-5
}

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005
classes = 20
bbox = 5

# load image generator
print("loading image generator...")


# load model
print("loading initial model...")

pretrained_model = Pretrained(classes=1000)

weights_path = "./weights/darknet19_448.weights"

with open(weights_path, "rb") as f:
    dat = np.fromfile(f, dtype=np.float32)[4:]
    offset = 0
    layers = [[3,32,3],[32,64,3],[64,128,3],[128, 64,1],[64,128,3],[128,256,3],[256,128,1],[128,256,3],
        [256,512,3],[512,256,1],[256,512,3],[512,256,1],[256,512,3],[512,1024,3],[1024,512,1],
        [512,1024,3],[1024,512,1],[512,1024,3]]

    for i, l in enumerate(layers):
        in_ch = l[0]
        out_ch = l[1]
        size = l[2]


        txt = "pretrained_model.conv%d.beta = rm.Variable(dat[%d:%d].reshape(1, %d, 1, 1))" % (i+1, offset, offset+out_ch, out_ch)
        offset += out_ch
        exec(txt)

        #TODO batch_normalizeのパラメタ
        txt = "pretrained_model.conv%d.gamma = rm.Variable(dat[%d:%d].reshape(1, %d, 1, 1))" % (i+1, offset, offset+out_ch, out_ch)
        offset += out_ch
        exec(txt)

        txt = "pretrained_model.conv%d.bn._mov_mean = rm.Variable(dat[%d:%d])" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        txt = "pretrained_model.conv%d.bn._mov_std = rm.Variable(np.sqrt(dat[%d:%d]))" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        txt = "pretrained_model.conv%d.conv.params['w'] = rm.Variable(dat[%d:%d].reshape(%d,%d,%d,%d))" % (i+1, offset, offset+(out_ch*in_ch*size*size), out_ch, in_ch, size, size)
        offset += out_ch*in_ch*size*size
        exec(txt)

        txt = "pretrained_model.conv%d.conv.params['b'] = rm.Variable(np.zeros((1, %d, 1, 1)))" % (i+1, out_ch)
        exec(txt)

        print(i, offset)

    in_ch = 1024
    out_ch = 1000

    txt= "pretrained_model.conv23.params['b'] = rm.Variable(dat[%d:%d])" % (offset, offset+out_ch)
    offset += out_ch
    exec(txt)

    txt= "pretrained_model.conv23.params['w'] = rm.Variable(dat[%d:%d].reshape(%d,%d,1,1))" % (offset, offset+out_ch*in_ch*1, out_ch,in_ch)
    offset += out_ch*in_ch*1
    exec(txt)

    print(dat.shape[0] - offset)


yolo_model = YOLOv2(classes=classes, bbox=bbox)

opt = Sgd(lr=learning_rate, momentum=momentum)
voc = VOCgenerator()

# start to train
print("start training")
for epoch in range(epochs):
    image_generator = voc.generate_samples(batch_size=batch_size)
    batch = 0
    if str(epoch) in learning_schedules:
        opt._lr = learning_schedules[str(epoch)]
    while True:
        try:
            x, t = next(image_generator)
        except StopIteration:
            break
        batch += 1
        loss = yolo_train(yolo_model, pretrained_model, x, t, opt, weight_decay)
        print("epoch: %d    batch: %d   learning rate: %f   loss: %f" % (epoch+1, batch+1, opt._lr, loss))
        print("/////////////////////////////////////")

    # save model
    if (epoch + 1) % 10 ==0:
        model_file = "%s/%s.h5" % (backup_path, epoch+1)
        print("saving model to %s" % (model_file))
        yolo_model.save(model_file)
        yolo_model.save(backup_file)

print("saving model to %s/yolov2_final.h5" % (backup_path))
yolo_model.save("%s/yolov2_final.h5" % (backup_path))
