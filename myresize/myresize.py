
"""
COMP590 Assignment1
Linear Interpolation
Qiuyang Zhang
"""


import numpy as np
#import cv2
#import matplotlib.pyplot as plt
import math

def nn_resize(im, dim):
    # TODO Fill in 
    num_cols, num_rows = dim
    num_chans= im.shape[2]
    
    #know the scale
    row_ratio =  num_rows/im.shape[0]
    col_ratio = num_cols/im.shape[1]
    
    #create the new image
    new_shape = (num_rows,num_cols,num_chans)
    new_img = np.ones(new_shape,dtype = int)
    for i in range(num_rows):
        for j in range(num_cols):
            new_img[i,j] = (im[math.floor(i/row_ratio), math.floor(j/col_ratio)]).astype(int)
            
    return new_img
            

def bilinear_resize(im, dim):
    # TODO Fill in     
    num_cols, num_rows = dim
    num_chans= im.shape[2]
    
    #create a zero_padding image
    zeropadding_im_shape = (im.shape[0]+2,im.shape[1]+2,im.shape[2])
    zeropadding_im = np.zeros(zeropadding_im_shape,dtype=int)
    zeropadding_im[:,:,:] = 255
    zeropadding_im[1:im.shape[0]+1,1:im.shape[1]+1] = im
    
    #know the scale
    row_scale =  num_rows/im.shape[0]
    col_scale = num_cols/im.shape[1]
    
    #create a new image
    destination_shape = (num_rows,num_cols,num_chans)
    destination_img = np.zeros(destination_shape,dtype = int)
    
    #create two points with same num of channels
    q0 = np.zeros(num_chans,dtype=int)
    q1 = np.zeros(num_chans,dtype=int)
    
    for destination_x in range(num_rows):
        for destination_y in range(num_cols):
            for n in range(num_chans):
                source_x = (destination_x + 0.5) * (1/row_scale) - 0.5
                source_y = (destination_y + 0.5) * (1/col_scale) - 0.5
                
                source_x0 = math.floor(source_x)
                source_x1 = math.ceil(source_x)
                source_y0 = math.floor(source_y)
                source_y1 = math.ceil(source_y)

                d_x0 = abs(source_x - source_x0)
                d_x1 = abs(source_x - source_x1)
                d_y0 = abs(source_y - source_y0)
                d_y1 = abs(source_y - source_y1)
                
                
                q0[n] = zeropadding_im[source_x0,source_y0,n] * d_x1 + zeropadding_im[source_x1,source_y0,n] * d_x0
                q1[n] = zeropadding_im[source_x0,source_y1,n] * d_x1 + zeropadding_im[source_x1,source_y1,n] * d_x0
                
                destination_img[destination_x,destination_y,n] = q0[n] * d_y1 + q1[n] * d_y0
    return destination_img


