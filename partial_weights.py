import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from darknet19 import *
from yolov2 import *
from yolov2_grid_prob import *
from yolov2_bbox import *

classes = 10
bbox = 5
partial_layer = 18

def copy_conv_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.conv%d" % i)
        dst_layer = eval("dst.conv%d" % i)
        dst_layer.params = src_layer.params

# def copy_bias_layer(src, dst, layers):
#     for i in layers:
#         src_layer = eval("src.bias%d" % i)
#         dst_layer = eval("dst.bias%d" % i)
#         dst_layer.b = src_layer.b

# def copy_bn_layer(src, dst, layers):
#     for i in layers:
#         src_layer = eval("src.bn%d" % i)
#         dst_layer = eval("dst.bn%d" % i)
#         dst_layer.N = src_layer.N
#         dst_layer.avg_var = src_layer.avg_var
#         dst_layer.avg_mean = src_layer.avg_mean
#         dst_layer.gamma = src_layer.gamma
#         dst_layer.eps = src_layer.eps

# load model
print("loading original model...")
input_weight_file = "./backup/darknet19_448_final.h5"
output_weight_file = "./backup/partial.h5"

model = Darknet19(classes=classes)
model.load(input_weight_file) # load saved model

yolov2 = YOLOv2(classes=classes, bbox=bbox)
copy_conv_layer(model, yolov2, range(1, partial_layer+1))
#copy_bias_layer(model.predictor, yolov2, range(1, partial_layer+1))
#copy_bn_layer(model.predictor, yolov2, range(1, partial_layer+1))
model = YOLOv2Predictor(yolov2)

print("saving model to %s" % (output_weight_file))
model.save(output_weight_file)
