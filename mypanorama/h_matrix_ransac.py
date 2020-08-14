# <Your name>
# COMP 776, Fall 2017
# Assignment: RANSAC

import numpy as np
from random import sample
from h_matrix import estimate_homography, compute_h_matrix_inliers

# for purposes of experimentation, we'll keep the sampling fixed every time the
# program is run
np.random.seed(0)


#-------------------------------------------------------------------------------

# Run RANSAC on a dataset for a given model type
#
# @param data                 M x K numpy array containing all M observations in
#                             the dataset, each with K dimensions
# @param inlier_threshold     given some error function for the model (e.g.,
#                             point-to-plane distance if the model is a 3D
#                             plane), label input data as inliers if their error
#                             is smaller than this threshold
# @param confidence_threshold our chosen value p that determines the minimum
#                             confidence required to stop RANSAC
# @param max_num_trials       initial maximum number of RANSAC iterations N
#
# @return best_model          the associated best model for the inliers
# @return inlier_mask         length M numpy boolean array with True indicating
#                             that a data point is an inlier and False
#                             otherwise; inliers can be recovered by taking
#                             data[inlier_mask]
def ransac(data, inlier_threshold, confidence_threshold, max_num_trials):
    max_iter = max_num_trials # current maximum number of trials
    iter_count = 0            # current number of iterations

    S = 4

    best_inlier_count = 0          # initial number of inliers is zero
    best_inlier_mask = np.zeros(   # initially mark all samples as outliers
        len(data), dtype=np.bool)
    best_H = np.zeros((3, 3))      # dummy initial model

    #this is pairwise matching, so the keypoints from the first image is data[:,:2] and keypoints from the second image is data[:,2:]
    keypoints1 = np.column_stack((data[:,:2], np.ones(len(data))))
    keypoints2 = np.column_stack((data[:,2:], np.ones(len(data))))
    
    
    #initialize outlier ratio
    outlier_ratio = 0
    #--------------------------------------------------------------------------
    #
    # TODO: fill in the steps of F-matrix RANSAC, below
    #
    #--------------------------------------------------------------------------


    # continue while the maximum number of iterations hasn't been reached
    while iter_count < max_iter:
        iter_count += 1

        #-----------------------------------------------------------------------
        # 1) sample as many points from the data as are needed to fit the
        #    relevant model
        mask = sample(range(len(data)),S)
        #correspondences = np.zeros([4,4])
        correspondenes = data[mask]

        #-----------------------------------------------------------------------
        # 2) fit a model to the sampled data subset
        H = estimate_homography(correspondences)

        #-----------------------------------------------------------------------
        # 3) determine the inliers to the model; store the result as a boolean
        #    mask, with inliers referenced by data[inlier_mask]
        inlier_mask = compute_h_matrix_inliers(keypoints1, keypoints2, H, inlier_threshold)
        inlier_count = np.nonzero(inlier_mask)
        outlier_ratio = 1 - (inlier_count / len(data))
        #-----------------------------------------------------------------------
        # 4) if this model is the best one yet, update the report and the
        #    maximum iteration threshold
        print("inlier_count", inlier_count)
        if(inlier_count>best_inlier_count):
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_H = H
            max_iter = np.log(1-confidence_threshold) / np.log(1-(1-outlier_ratio)**S)
            print("max_iter", max_iter)
        

    #---------------------------------------------------------------------------
    # 5) run a final fit on the H matrix using the inliers


    #---------------------------------------------------------------------------
    # print some information about the results of RANSAC

    inlier_ratio = best_inlier_count / len(data)

    print("Iterations:", iter_count)
    print("Inlier Ratio: {:.3f}".format(inlier_ratio))
    print("Best Fit Model:")
    print("Best inlier count:", best_inlier_count)
    print("  [ {:7.4f}  {:7.4f}  {:7.4f} ]".format(*best_H[0]))
    print("  [ {:7.4f}  {:7.4f}  {:7.4f} ]".format(*best_H[1]))
    print("  [ {:7.4f}  {:7.4f}  {:7.4f} ]".format(*best_H[2]))

    return best_F, best_inlier_mask

