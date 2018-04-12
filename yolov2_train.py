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
from lib.utils import *
from lib.image_generator import *

set_cuda_active(True)

# hyper parameters
train_sizes = [320, 352, 384, 416]
item_path = "./items"
background_path = "./backgrounds"
backup_path = "backup"
backup_file = "%s/backup.h5" % (backup_path)
pretrained_weight_file = "%s/darknet19_448_final.h5" % (backup_path)
initial_weight_file = "%s/backup.h5" % (backup_path)
batch_size = 32
max_batches = 60000
learning_rate = 1e-5
learning_schedules = {
    "0"    : 1e-5,
    "1000"  : 1e-4,
    "20000": 1e-5,
    "40000": 1e-6
}

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005
classes = 10
bbox = 5

# load image generator
print("loading image generator...")
generator = ImageGenerator(item_path, background_path)

# load model
print("loading initial model...")
pretrained_model = Pretrained(classes=classes)
yolo_model = YOLOv2(classes=classes, bbox=bbox)
pretrained_model.load(pretrained_weight_file)
yolo_model.load(initial_weight_file)
#num_gpu = cuGetDeviceCount()

#model.to_gpu()

opt = Sgd(lr=learning_rate, momentum=momentum)

#trainer = Trainer(model, batch_size=32, loss_func=yolo_detector, num_epoch=1, optimizer=opt, num_gpu=num_gpu)

#input_width=input_height=320
#x, t = generator.generate_samples(
#        n_samples=16,
#        n_items=3,
#        crop_width=input_width,
#        crop_height=input_height,
#        min_item_scale=0.1,
#        max_item_scale=0.4,
#        rand_angle=25,
#        minimum_crop=0.8,
#        delta_hue=0.01,
#        delta_sat_scale=0.5,
#        delta_val_scale=0.5
#)
# start to train
print("start training")
for batch in range(max_batches):
    if str(batch) in learning_schedules:
        opt._lr = learning_schedules[str(batch)]
    if batch % 80 == 0:
       input_width = input_height = train_sizes[np.random.randint(len(train_sizes))]


    x, t = generator.generate_samples(
        n_samples=batch_size,
        n_items=3,
        crop_width=input_width,
        crop_height=input_height,
        min_item_scale=0.1,
        max_item_scale=0.2,
        rand_angle=25,
        minimum_crop=0.8,
        delta_hue=0.01,
        delta_sat_scale=0.5,
        delta_val_scale=0.5
    )
    # generate sample
    #x = Variable(x)
    #x.to_gpu()
    # forward
    loss = yolo_train(yolo_model, pretrained_model, x, t, opt, weight_decay)
    #print(model.conv22.params)
    #trainer.train(train_distributor=NdarrayDistributor(x, t))
    print("batch: %d     input size: %dx%d     learning rate: %f    loss: %f" % (batch, input_height, input_width, opt._lr, loss))
    print("/////////////////////////////////////")

    # save model
    if (batch+1) % 500 == 0:
        model_file = "%s/%s.h5" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        yolo_model.save(model_file)
        yolo_model.save(backup_file)

print("saving model to %s/yolov2_final.h5" % (backup_path))
yolo_model.save("%s/yolov2_final.h5" % (backup_path))

yolo_model.to_cpu()
yolo_model.save("%s/yolov2_final_cpu.h5" % (backup_path))
