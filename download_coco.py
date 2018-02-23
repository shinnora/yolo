from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
from lib.utils import *
import matplotlib.pyplot as plt

class CocoGenerator():

    def __init__(self, dataDir='../data/', dataType='val2017'):
        self.annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
        self.coco = COCO(self.annFile)
        self.catNum = len(self.coco.getCatIds())
        self.imgIds = self.coco.getImgIds()

    def generate_samples(self, batch_size, width, height):
        imgorder = np.random.permutation(self.imgIds)
        length = len(order)
        for imgId in imgorder:
            img = self.coco.loadImgs(imgId)[0]
            I = reshape_to_yolo_size(io.imread(img['coco_url']))
            anns = self.coco.loadAnns(imgIds = self.coco.getAnnIds(imgId))
            for ann in anns:
                
