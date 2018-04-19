import numpy as np
import cv2
import os
import sys
from renom.utility.image.data_augmentation import *
from xml.etree import ElementTree
from itertools import product
import urllib.request as request


class VOCgenerator():

    def __init__(self, ann_path="VOCdevkit/VOC2012/Annotations/", img_path="VOCdevkit/VOC2012/JPEGImages/"):
        self.ann_path = ann_path
        self.img_path = img_path
        self.train_file_list = [path for path in sorted(os.listdir(ann_path)) if not "2012_" in path]
        self.test_file_list = [path for path in os.listdir(ann_path) if "2012_" in path]
        self.cat_num = 20
        self.label_dict = {}

    def one_hot(self, id):
        one_hot_label = np.zeros(self.cat_num)
        one_hot_label[id] = 1
        return one_hot_label

    def generate_samples(self, batch_size, size, train=True):
        if train:
            file_list = self.train_file_list
        else:
            file_list = self.test_file_list
        img_order = np.random.permutation(len(file_list))
        for batch_num in range(int(len(img_order) / batch_size)):
            x = []
            t = []
            augmenter = DataAugmentation([Color_jitter(s=(1.0,1.5), v=(1.0,1.5)),Resize(size=size)])
            for img_num in img_order[batch_num * batch_size : (batch_num + 1) * batch_size]:
                file_name, ground_truths = self._get_img_info(file_list[img_num])
                img_path = os.path.join(self.img_path, file_name)
                img = augmenter.create(np.asarray(cv2.imread(img_path), dtype=np.float32)) / 255.0
                img = img.transpose(2, 0, 1)
                x.append(img)
                t.append(ground_truths)
            yield x, t

    def _get_obj_coordinate(self, obj):
        class_name = obj.find("name").text.strip()
        if self.label_dict.get(class_name, None) is None:
            self.label_dict[class_name] = len(self.label_dict)
        class_id = self.label_dict[class_name]
        bbox = obj.find("bndbox")
        xmax = float(bbox.find("xmax").text.strip())
        xmin = float(bbox.find("xmin").text.strip())
        ymax = float(bbox.find("ymax").text.strip())
        ymin = float(bbox.find("ymin").text.strip())
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + w/2
        y = ymin + h/2
        return class_id, x, y, w, h

    def _get_img_info(self, filename):
        tree = ElementTree.parse(os.path.join(self.ann_path, filename))
        node = tree.getroot()
        file_name = node.find("filename").text.strip()
        img_h = float(node.find("size").find("height").text.strip())
        img_w = float(node.find("size").find("width").text.strip())
        obj_list = node.findall("object")
        ground_truths = []
        for obj in obj_list:
            class_id, x, y, w, h = self._get_obj_coordinate(obj)
            ground_truths.append({
                "x" : x / img_w,
                "y" : y / img_h,
                "w" : w / img_w,
                "h" : h / img_h,
                "label" : class_id,
                "one_hot_label" : self.one_hot(class_id)
            })
        return file_name, ground_truths


# x, y, w, hの4パラメータを保持するだけのクラス
class Box():
    def __init__(self, x, y, w, h):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)

    def int_left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(float(self.x - half_width))), int(round(float(self.y - half_height))))

    def left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x - half_width, self.y - half_height]

    def int_right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(float(self.x + half_width))), int(round(float(self.y + half_height))))

    def right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x + half_width, self.y + half_height]

    def crop_region(self, h, w):
        left, top = self.left_top()
        right, bottom = self.right_bottom()
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        self.w = right - left
        self.h = bottom - top
        self.x = (right + left) / 2
        self.y = (bottom + top) / 2
        return self

# 2本の線の情報を受取り、被ってる線分の長さを返す。あくまで線分
def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

# chainerのVariable用のoverlap
def multi_overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = np.maximum(x1 - len1_half, x2 - len2_half)
    right = np.minimum(x1 + len1_half, x2 + len2_half)

    return right - left

# 2つのboxを受け取り、被ってる面積を返す(intersection of 2 boxes)
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

# chainer用
def multi_box_intersection(a, b):
    w = multi_overlap(a.x, a.w, b.x, b.w)
    h = multi_overlap(a.y, a.h, b.y, b.h)
    zeros = np.zeros(w.shape, dtype=w.data.dtype)

    w = np.maximum(w, zeros)
    h = np.maximum(h, zeros)

    area = w * h
    return area

# 2つのboxを受け取り、合計面積を返す。(union of 2 boxes)
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# chianer用
def multi_box_union(a, b):
    i = multi_box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# compute iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

# chainer用
def multi_box_iou(a, b):
    return multi_box_intersection(a, b) / multi_box_union(a, b)


# 画像を読み込んで、hue, sat, val空間でランダム変換を加える関数
def random_hsv_image(bgr_image, delta_hue, delta_sat_scale, delta_val_scale):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # hue
    hsv_image[:, :, 0] += int((np.random.rand() * delta_hue * 2 - delta_hue) * 255)

    # sat
    sat_scale = 1 + np.random.rand() * delta_sat_scale * 2 - delta_sat_scale
    hsv_image[:, :, 1] *= sat_scale

    # val
    val_scale = 1 + np.random.rand() * delta_val_scale * 2 - delta_val_scale
    hsv_image[:, :, 2] *= val_scale

    hsv_image[hsv_image < 0] = 0
    hsv_image[hsv_image > 255] = 255
    hsv_image = hsv_image.astype(np.uint8)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image

# non maximum suppression
def nms(predicted_results, iou_thresh):
    nms_results = []
    for i in range(len(predicted_results)):
        overlapped = False
        for j in range(i+1, len(predicted_results)):
            if box_iou(predicted_results[i]["box"], predicted_results[j]["box"]) > iou_thresh:
                overlapped = True
                if predicted_results[i]["objectness"] > predicted_results[j]["objectness"]:
                    temp = predicted_results[i]
                    predicted_results[i] = predicted_results[j]
                    predicted_results[j] = temp
        if not overlapped:
            nms_results.append(predicted_results[i])
    return nms_results
