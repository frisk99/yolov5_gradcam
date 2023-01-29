import os
import time
import argparse
import numpy as np
from models.gradcam import YOLOV5GradCAM
from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
import cv2
import torch
from deep_utils import Box, split_extension
A = torch.tensor([[[1],
        		  [4],
        		  [7,]]])
print("A的维度为{}" .format(A.dim()))
print("A形状为{}" .format(A.shape))
B = torch.zeros(1,10,1)
C=torch.cat((A,B),1)
print(C)
