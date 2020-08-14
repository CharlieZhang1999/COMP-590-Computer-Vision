# Qiuyang Zhang
# COMP 590, Spring 2020
# Assignment: Feature Extraction

import numpy as np
from myfiltering import cross_correlation_2d, convolve_2d
import math
#-------------------------------------------------------------------------------

class FeatureMatcher:
    def __init__(self, args):
        self.window_size = args.matching_window_size
        if self.window_size % 2 != 1:
            raise ValueError("window size must be an odd number")

        if args.matching_method.lower() == "ssd":
            self.matching_method = self.match_ssd
        elif args.matching_method.lower() == "ncc":
            self.matching_method = self.match_ncc
        else:
            raise ValueError("invalid matching method")

    #---------------------------------------------------------------------------

    # extract descriptors and match keypoints between two images
    #
    # inputs:
    # - image1: first input image, assumed to be grayscale
    # - keypoints1: N1 x 2 array of keypoint (x,y) pixel locations in the first
    #   image, assumed to be integer coordinates
    # - image2: second input image, assumed to be grayscale
    # - keypoints2: N2 x 2 array of keypoint (x,y) pixel locations in the second
    #   image, assumed to be integer coordinates
    # returns:
    # - matches: M x 2 array of indices for the matches; the first column
    #   provides the index for the keypoint in the first image, and the second
    #   column provides the corresponding keypoint index in the second image
    def __call__(self, image1, keypoints1, image2, keypoints2):
        d1 = self.get_descriptors(image1, keypoints1)
        d2 = self.get_descriptors(image2, keypoints2)

        match_matrix = self.matching_method(d1, d2)
        matches = self.compute_matches(match_matrix)

        return matches

    #---------------------------------------------------------------------------

    # extract descriptors from an image
    #
    # inputs:
    # - image: input image, assumed to be grayscale
    # - keypoints: N x 2 array of keypoint (x,y) pixel locations in the image,
    #   assumed to be integer coordinates
    # returns:
    # - descriptors: N x <window size**2> array of feature descriptors for the
    #   keypoints; in the implementation here, the descriptors are the
    #   (window_size, window_size) patch centered at every keypoint
    def get_descriptors(self, image, keypoints):
        # TODO: Obtain descriptors for the provided keypoints
        
        #define output
        output = []
        
        #define N, the number of keypoints
        N = keypoints.shape[0]
        
        
        #define half window size
        half_width = math.floor(self.window_size/2)
        
        #zero padding
        #zero_matrix = np.zeros((self.window_size,self.window_size))
        #zero_matrix[half_width] = 1
        #new_im = convolve_2d(image, zero_matrix, 'full')

        #get x-coordinate and y-coordinate in the key point and their descriptor in ravelled form
        for coor in keypoints:
            x_coor = coor[0]
            y_coor = coor[1]
            output.append(image[y_coor - half_width:y_coor + half_width+1, x_coor - half_width:x_coor + half_width+1])
        output = np.array(output)    
        output = output.reshape((N, -1))
            
        return output

    #---------------------------------------------------------------------------

    # compute a distance matrix between two sets of feature descriptors using
    # sum-of-squares differences
    #
    # inputs:
    # - d1: N1 x <feature_length> array of keypoint descriptors
    # - d2: N2 x <feature_length> array of keypoint descriptors
    # returns:
    # - match_matrix: N1 x N2 array of descriptor distances, with the rows
    #   corresponding to d1 and the columns corresponding to d2
    def match_ssd(self, d1, d2):
        # TODO: implement SSD comparison of the two descriptor sets
        output = np.zeros((d1.shape[0], d2.shape[0]))
        for d_1 in range(d1.shape[0]):
            for d_2 in range(d2.shape[0]):
                output[d_1,d_2] = np.sum((d1[d_1] - d2[d_2])**2)
        return output

    #---------------------------------------------------------------------------

    # compute a distance matrix between two sets of feature descriptors using
    # one minus the normalized cross-correlation
    #
    # inputs:
    # - d1: N1 x <feature_length> array of keypoint descriptors
    # - d2: N2 x <feature_length> array of keypoint descriptors
    # returns:
    # - match_matrix: N1 x N2 array of descriptor distances, with the rows
    #   corresponding to d1 and the columns corresponding to d2
    def match_ncc(self, d1, d2):
        # TODO: implement 1 - NCC comparison of the two descriptor sets
        ncc_output = np.zeros((d1.shape[0], d2.shape[0]))
        for d_1 in range(d1.shape[0]):
            for d_2 in range(d2.shape[0]):
                normalized_d1 = (d1[d_1] - np.mean(d1[d_1])) / np.std(d1[d_1])
                normalized_d2 = (d2[d_2] - np.mean(d2[d_2])) / np.std(d2[d_2])
                ncc_output[d_1,d_2] = 1 - np.sum(normalized_d1 * normalized_d2) #this is 1-ncc
        return ncc_output

    #---------------------------------------------------------------------------

    # given a matrix of descriptor distances for keypoint pairs, compute
    # keypoint correspondences between two images
    #
    # inputs:
    # - match_matrix: N1 x N2 array of descriptor distances, with the rows
    #   corresponding to the N1 keypoints in the first image and the columns
    #   corresponding to the N2 keypoints in the second image
    # returns:
    # - matches: M x 2 array of indices for the M matches; the first column
    #   provides the index for the keypoint in the first image, and the second
    #   column provides the corresponding keypoint index in the second image
    def compute_matches(self, match_matrix):
        # TODO: implement one-to-one matching for the match matrix
        matching_output = []
        for row in range(match_matrix.shape[0]):
            for col in range(match_matrix.shape[1]):
                if (col == np.argmin(match_matrix[row,:])) & (row == np.argmin(match_matrix[:,col])):
                    arr = np.array([row, col])
                    matching_output.append(arr)
                    
        matching_output = np.array(matching_output)
        matching_output = matching_output.reshape((-1, 2))
        return matching_output
    

        