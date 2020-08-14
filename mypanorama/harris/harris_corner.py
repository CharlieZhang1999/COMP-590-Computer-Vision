# <Your name>
# COMP 776, Fall 2017
# Assignment: Feature Extraction

import numpy as np

from scipy.ndimage.filters import gaussian_filter, maximum_filter, sobel

#-------------------------------------------------------------------------------

class HarrisCornerFeatureDetector:
    def __init__(self, args):
        self.gaussian_sigma = args.gaussian_sigma
        self.maxfilter_window_size = args.maxfilter_window_size
        self.harris_corner_k = args.harris_corner_k
        self.response_threshold = args.response_threshold
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
        # compute image gradients using the Sobel operator; we'll normalize
        # (divide by 8) to obtain correct units for the derivative, but note
        # that the algorithm will perform equally well without normalization
        I_y = sobel(image, axis=0)
        I_x = sobel(image, axis=1)

        A = gaussian_filter(I_x**2, self.gaussian_sigma)
        B = gaussian_filter(I_y**2, self.gaussian_sigma)
        C = gaussian_filter(I_x * I_y, self.gaussian_sigma)

        # Harris corner response
        R = A * B - C**2 - self.harris_corner_k * (A + B)**2

        # Noble's corner response
        #R = 2. * (A * B - C**2) / (A + B + np.finfo("float").eps)

        return R

    #---------------------------------------------------------------------------

    # find (x,y) pixel coordinates of maxima in a corner response map
    # inputs:
    # - R: Harris corner response map
    # returns:
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the corner
    #   response map, assumed to be integer coordinates
    def get_keypoints(self, R):
        # non-maxima suppression
        R_max = maximum_filter(R, size=self.maxfilter_window_size)

        mask = (R == R_max)
        threshold = self.response_threshold * max(R.max(), 0.)
        mask[R_max<threshold] = 0
        coords = np.where(mask) # separate arrays of y and x coordinates
        keypoints = np.column_stack(coords[::-1]) # Nx2 array of (x, y) coords

        if self.max_num_features is not None:
            idxs = np.argsort(R[coords])[-self.max_num_features:]
            keypoints = keypoints[idxs]

        return keypoints
