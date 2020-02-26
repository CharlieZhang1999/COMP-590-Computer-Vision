#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:48:35 2020

@author: djogem
"""

from myresize import *
import cv2
import matplotlib.pyplot as plt 
im = cv2.imread("/Users/djogem/Downloads/COMP 590-153/HW1/dogsmall.jpg")


dim = (im.shape[1]*4, im.shape[0]*4)
a = nn_resize(im, dim)
plt.imshow(a)
plt.show()

'''
import math
import numpy as np
print(math.ceil(5))

a = np.zeros(3)
print(a)
'''

