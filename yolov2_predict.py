#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import matplotlib.pyplot as plt
import copy
import time
import numpy as np
import argparse
import cv2

from lib.utils import *

import renom as rm
from renom.cuda import cuda
from renom.optimizer import Sgd
from renom.core import DEBUG_NODE_STAT, DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH

from yolov2 import *
from darknet19 import *

class AnimalPredictor:

    def __init__(self):
        weight_file = "./backup/yolov2_final_cpu.h5"
        self.classes = 10
        self.bbox = 5
        self.detection_thresh = 0.3
        self.iou_thresh = 0.3
        self.label_file = "./data/label.txt"
        with open(self.label_file, "r") as f:
            self.labels = f.read().strip.split("\n")

        model = YOLOv2(classes=self.classes, bbox=self.bbox)
        model.load(weight_file)
        self.model = model

    def __call__(self, orig_img):
        orig_input_height, orig_input_width, _ = orig_img.shape
        img = reshape_to_yolo_size(orig_img)
        input_height, input_width, _ = img.shape
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        x_data = img[np.newaxis, :, :, :]
        x = rm.Variable(x_data)
        x, y, w, h, conf, prob = yolo_predict(model, x)
        _, _, _, grid_h, grid_w = x.shape
        x = np.reshape(x, (self.bbox, grid_h, grid_w))
        y = np.reshape(y, (self.bbox, grid_h, grid_w))
        w = np.reshape(w, (self.bbox, grid_h, grid_w))
        h = np.reshape(h, (self.bbox, grid_h, grid_w))
        conf = np.reshape(conf, (self.bbox, grid_h, grid_w))
        prob = np.transpose(np.reshape(prob, (self.bbox, self.classes, grid_h, grid_w)), (1, 0, 2, 3))
        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh

        results = []
        for i in range(detected_indices.sum()):
            results.append({
                "label": self.labels[prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()],
                "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
                "conf" : conf[detected_indices][i],
                "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
                "box"  : Box(
                            x[detected_indices][i]*orig_input_width,
                            y[detected_indices][i]*orig_input_height,
                            w[detected_indices][i]*orig_input_width,
                            h[detected_indices][i]*orig_input_height).crop_region(orig_input_height, orig_input_width)
            })

        # nms
        nms_results = nms(results, self.iou_thresh)
        return nms_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="指定したパスの画像を読み込み、bbox及びクラスの予測を行う")
    parser.add_argument('path', help="画像ファイルへのパスを指定")
    args = parser.parse_args()
    image_file = args.path

    # read image
    print("loading image...")
    orig_img = cv2.imread(image_file)

    predictor = AnimalPredictor()
    nms_results = predictor(orig_img)

    # draw result
    for result in nms_results:
        left, top = result["box"].int_left_top()
        cv2.rectangle(
            orig_img,
            result["box"].int_left_top(), result["box"].int_right_bottom(),
            (255, 0, 255),
            3
        )
        text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
        cv2.putText(orig_img, text, (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(text)
