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
train_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
backup_path = "./weights"
backup_file = "%s/backup.h5" % (backup_path)
pretrained_weight_file = "%s/darknet19_448.h5" % (backup_path)
initial_weight_file = "%s/backup.h5" % (backup_path)
batch_size = 32
epochs = 160
learning_rate = 1e-3
learning_schedules = {
    "0" : 1e-3,
    "60": 1e-4,
    "90": 1e-5
}

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005
classes = 20
bbox = 5

# load image generator
print("loading image generator...")
generator = VOCgenerator()

# load model
print("loading initial model...")
pretrained_model = Pretrained(classes=classes)
yolo_model = YOLOv2(classes=classes, bbox=bbox)
pretrained_model.load(pretrained_weight_file)

opt = Sgd(lr=learning_rate, momentum=momentum)

# start to train
print("start training")
for epoch in range(epochs):
    batch = 0
    if str(epoch) in learning_schedules:
        opt._lr = learning_schedules[str(epoch)]
    while True:
        if batch % 10 == 0:
           train_size = train_sizes[np.random.randint(len(train_sizes))]
        try:
            x, t = generator.generate_samples(
                batch_size=batch_size,
                size=train_size
            )
            batch += 1
        except StopIteration:
            break
        loss = yolo_train(yolo_model, pretrained_model, x, t, opt, weight_decay)
        print("epoch: %d    batch: %d     input size: %dx%d     learning rate: %f    loss: %f" % (epoch, batch, train_size, train_size, opt._lr, loss))
        print("/////////////////////////////////////")

    # save model
    if (epoch + 1) % 10 ==0:
        model_file = "%s/%s.h5" % (backup_path, epoch+1)
        print("saving model to %s" % (model_file))
        yolo_model.save(model_file)
        yolo_model.save(backup_file)

print("saving model to %s/yolov2_final.h5" % (backup_path))
yolo_model.save("%s/yolov2_final.h5" % (backup_path))
