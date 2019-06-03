#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:15:37 2019

@author: uesr
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
output_path = '/home/uesr/Desktop/Mask_RCNN/Mask_RCNN/Data/final/image_processed/'
path  = glob.glob("/home/uesr/Desktop/Mask_RCNN/Mask_RCNN/Data/final/images/*.png")
for i in range(len(path)):
    x = path[i]
    src_fname, ext = os.path.splitext(x) 
    img= Image.open(x)
    """
    Some preprocessing step
    """
    save_fname = os.path.join(output_path, os.path.basename(src_fname)+'.png')
    img.save(save_fname)
    