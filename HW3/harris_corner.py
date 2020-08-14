# Qiuyang Zhang
# COMP 590, Spring 2020
# Assignment: Feature Extraction

import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter, maximum_filter, sobel
from myfiltering import cross_correlation_2d, convolve_2d, gaussian_blur_kernel_2d, sobel_kernel, sobel_image

#-------------------------------------------------------------------------------

class HarrisCornerFeatureDetector:
    def __init__(self, args):
        self.gaussian_sigma = args.gaussian_sigma
        self.maxfilter_window_size = args.maxfilter_window_size
        self.harris_corner_k = args.harris_corner_k
        self.max_num_features = args.max_num_features

    #---------------------------------------------------------------------------

    # detect corner features in an input image
    # inputs:
    # - image: a grayscale image
    # returns:
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the image,
    #   assumed to be integer coordinates
    def __call__(self, image):
        corner_response = self.compute_corner_response(image)
        keypoints = self.get_keypoints(corner_response)

        return keypoints

    #---------------------------------------------------------------------------

    # compute the Harris corner response function for each point in the image
    #   R(x, y) = det(M(x, y) - k * tr(M(x, y))^2
    # where
    #             [      I_x(x, y)^2        I_x(x, y) * I_y(x, y) ]
    #   M(x, y) = [ I_x(x, y) * I_y(x, y)        I_y(x, y)^2      ] * G
    #
    # with "* G" denoting convolution with a 2D Gaussian.
    #
    # inputs:
    # - image: a grayscale image
    # returns:
    # - R: transformation of the input image to reflect "cornerness"
    def compute_corner_response(self, image):
        # TODO: Compute the Harris corner response
        R = np.zeros_like(image)
        
        
        #sobel kernel
        Ix = sobel(image, axis=1)
        Iy = sobel(image, axis=0)

        Ix_2 = Ix**2
        Ix_Iy = Ix * Iy
        Iy_2 = Iy**2
        
        
        #window_size and gaussian filter(used for weighting)
        win_size = 7
        half_width = math.floor(win_size/2)
       
        #get row and col for for-loop
        row, col = R.shape
        
        #initialize k
        k = self.harris_corner_k
        
        #initialize gaussian_sigma
        gaussian_sigma = self.gaussian_sigma
        
        #get R(x,y)
        for i in range(half_width, row-half_width):
            for j in range(half_width, col-half_width):
                
                A = np.sum(gaussian_filter(Ix_2[i-half_width:i+half_width+1, j-half_width:j+half_width+1], gaussian_sigma))
                B = np.sum(gaussian_filter(Ix_Iy[i-half_width:i+half_width+1, j-half_width:j+half_width+1], gaussian_sigma))
                C = np.sum(gaussian_filter(Iy_2[i-half_width:i+half_width+1, j-half_width:j+half_width+1], gaussian_sigma))
                M = np.array([[A,B],[B,C]])
                #M = np.array([[new_Ix_2[i,j],new_Ix_Iy[i,j]],[new_Ix_Iy[i,j],new_Iy_2[i,j]]])
                det = np.linalg.det(M)
                trace = np.trace(M)
                R[i,j] = det - k*(trace**2)
        
        return R

    #---------------------------------------------------------------------------

    # find (x,y) pixel coordinates of maxima in a corner response map
    # inputs:
    # - R: Harris corner response map
    # returns:
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the corner
    #   response map, assumed to be integer coordinates
    def get_keypoints(self, R):
        # TODO: apply non-maximum suppression and obtain the up-to-K strongest
        # features having R(x,y) > 0
        
        w = self.maxfilter_window_size
                
        local_max = maximum_filter(R,size=(w,w))
        local_max[local_max != R] = 0
        
        
        reverse_sort = np.argsort(local_max.ravel())
        sort  = reverse_sort[::-1]
        K = self.max_num_features
        y_coor, x_coor = np.unravel_index(sort[:K], R.shape)
        coor = np.column_stack((x_coor,y_coor))
        
        return coor


  