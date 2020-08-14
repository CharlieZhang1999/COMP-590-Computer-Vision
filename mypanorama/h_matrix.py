# <Your name>
# COMP 776, Fall 2017
# Assignment: RANSAC

import numpy as np


#-------------------------------------------------------------------------------

# estimate a fundamental matrix from a set of correspondences
# here, we'll apply Hartley normalization, do a DLT, and enforce the rank-2
# constraint on the F matrix
# NOTE: Don't forget to pre-/post-multiply your final F matrix by the
#   Hartley transformations!
#
# @param correspondences      M x 4 numpy array of (x1, y1, x2, y2) matches
#
# @return F                   3x3 fundamental matrix
def estimate_homography(correspondences):

    # TODO
    # build the data matrix and solve using SVD
    H = np.ones(9)
    #build A matrix
    A = np.zeros((correspondences.shape[0]*2, 8))
    
    #build b matrix
    b = np.zeros(correspondences.shape[0]*2)
    
    for i in range(correspondences.shape[0]):
        A[i*2] = [correspondences[i,0],correspondences[i,1],1,0,0,0,-correspondences[i,0]*correspondences[i,2], -correspondences[i,1]*correspondences[i,2]]
        A[i*2+1] = [0,0,0,correspondences[i,0],correspondences[i,1],1,-correspondences[i,0]*correspondences[i,3],-correspondences[i,1]*correspondences[i,3]]
        b[i*2] = correspondences[i,2]
        b[i*2+1] = correspondences[i,3]
        
    #print("A", A)
    #print("b", b)
    ATA_inverse = np.linalg.inv(np.dot(A.T, A))
    ATB = np.dot(A.T,b)
    
    #H = (A^T A)^-1 (A^T b)
    H_temp = np.dot(ATA_inverse,ATB)
    #H = np.eye(3)
    
    H[:8] = H_temp
    H = H.reshape(3,3)
    print("these are the homography result between the correspondences")
    
    return H


#-------------------------------------------------------------------------------

# given a set of 2D keypoint correspondences in a pair of images, and a
# hypothesized homography matrix relating the two images, compute which
# correspondences are inliers:
# 
#
# @param keypoints1           M x 3 array of (x,y,1) coordinates (first image)
# @param keypoints2           M x 3 array of (x,y,1) coordinates (second image)
# @return H                   3x3 homography matrix hypothesis
# @param inlier_threshold     given some error function for the model (e.g.,
#                             point-to-point distance )
# @return inlier_mask         length M numpy boolean array with True indicating
#                             that a data point is an inlier and False
def compute_h_matrix_inliers(keypoints1, keypoints2, H, inlier_threshold):
    inlier_mask = np.zeros(len(keypoints1))
    
    keypoints1_t = keypoints1.T
    #get x',y', it's a 3 by len(keypoints1)  matrix
    result = np.dot(H,keypoints1_t)
    result[:2,:] = result[:2,:] / result[2, :]
    result[2, :] = 1
    
    #to compare them with M*3 array, has to transpose them back to M*3 array
    transformed_keypoints1 = result.T
    #if < threshold, 1; else, 0
    inlier_mask = (np.sum((transformed_keypoints1 - keypoints2)**2, 1)<inlier_threshold)
    
    return inlier_mask