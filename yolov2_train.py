import time
import cv2
import numpy as np
import glob
import os
import renom as rm
from renom.optimizer import Sgd
from renom.utility.trainer import Trainer
from renom.cuda.cuda import set_cuda_active, cuGetDeviceCount, cuDeviceSynchronize
from renom import cuda
from renom.utility.distributor import NdarrayDistributor
from yolov2 import *
from lib.utils import *
from lib.image_generator import *

set_cuda_active(True)

# hyper parameters
train_sizes = [320, 352, 384, 416, 448]
item_path = "./items"
background_path = "./backgrounds"
initial_weight_file = "./backup/partial.h5"
backup_path = "backup"
backup_file = "%s/backup.h5" % (backup_path)
batch_size = 16
max_batches = 30000
learning_rate = 1e-5
learning_schedules = {
    "0"    : 1e-5,
    "500"  : 1e-4,
    "10000": 1e-5,
    "20000": 1e-6
}

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.005
classes = 10
bbox = 5

# load image generator
print("loading image generator...")
generator = ImageGenerator(item_path, background_path)

# load model
print("loading initial model...")
model = YOLOv2(classes=classes, bbox=bbox)
model.load(initial_weight_file)
num_gpu = cuGetDeviceCount()

#model.to_gpu()

opt = Sgd(lr=learning_rate, momentum=momentum)

#trainer = Trainer(model, batch_size=32, loss_func=yolo_detector, num_epoch=1, optimizer=opt, num_gpu=num_gpu)

# start to train
print("start training")
for batch in range(max_batches):
    if str(batch) in learning_schedules:
        opt._lr = learning_schedules[str(batch)]
    if batch % 80 == 0:
        input_width = input_height = train_sizes[np.random.randint(len(train_sizes))]

    # generate sample
    x, t = generator.generate_samples(
        n_samples=16,
        n_items=3,
        crop_width=input_width,
        crop_height=input_height,
        min_item_scale=0.1,
        max_item_scale=0.4,
        rand_angle=25,
        minimum_crop=0.8,
        delta_hue=0.01,
        delta_sat_scale=0.5,
        delta_val_scale=0.5
    )
    #x = Variable(x)
    #x.to_gpu()
    # forward
    loss = yolo_train(model, x, t, opt)
    #trainer.train(train_distributor=NdarrayDistributor(x, t))
    print("batch: %d     input size: %dx%d     learning rate: %f    loss: %f" % (batch, input_height, input_width, opt._lr, loss))
    print("/////////////////////////////////////")

    # save model
    if (batch+1) % 500 == 0:
        model_file = "%s/%s.h5" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        model.save(model_file)
        model.save(backup_file)

print("saving model to %s/yolov2_final.h5" % (backup_path))
model.save("%s/yolov2_final.h5" % (backup_path))

model.to_cpu()
model.save("%s/yolov2_final_cpu.h5" % (backup_path))
