# <Your name>
# COMP 590, Spring 2020
# Assignment: Panorama

import matplotlib.pyplot as plt
import numpy as np
import math

from harris.feature_matcher import FeatureMatcher
from harris.harris_corner import HarrisCornerFeatureDetector
from harris.plot_util import plot_keypoints, plot_matches
from h_matrix import estimate_homography, compute_h_matrix_inliers
from h_matrix_ransac import ransac

# for purposes of experimentation, we'll keep the sampling fixed every time the
# program is run
np.random.seed(0)

#-------------------------------------------------------------------------------

# convert a color image to a grayscale image according to luminance
# this uses the conversion for modern CRT phosphors suggested by Poynton
# see: http://poynton.ca/PDFs/ColorFAQ.pdf
#
# input:
# - image: RGB uint8 image (values 0 to 255)
# returns:
# - gray_image: grayscale version of the input RGB image, with floating point
#   values between 0 and 1
def rgb2gray(image):
    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
    return (0.2125 * red + 0.7154 * green + 0.0721 * blue) / 255.

#-------------------------------------------------------------------------------
def bilinear_interpolation(source_y,source_x,image):
    #print("coordinate on image2 is", (source_y, source_x))
    source_x0 = int(np.floor(source_x))
    source_x1 = int(np.ceil(source_x))
    source_y0 = int(np.floor(source_y))
    source_y1 = int(np.ceil(source_y))

    d_x0 = abs(source_x - source_x0)
    d_x1 = abs(source_x - source_x1)
    d_y0 = abs(source_y - source_y0)
    d_y1 = abs(source_y - source_y1)

    q0 = image[source_y0,source_x0] * d_x1 + image[source_y0,source_x1] * d_x0
    q1 = image[source_y1,source_x0] * d_x1 + image[source_y1,source_x1]* d_x0
    
    #print("four points are", image[source_y0,source_x0],image[source_y0,source_x1], image[source_y1,source_x0], image[source_y1,source_x1])
    
    return q0*d_y1 + q1*d_y0


def main(args):
    # create the feature extractor
    extractor = HarrisCornerFeatureDetector(args)

    # convert uint8 RGB images to grayscale images in the range [0, 1]
    #modified
    image1 = plt.imread(args[0])
    image1 = rgb2gray(image1)
    
    #initialize 
    b_top_left_x = np.zeros(len(args))
    b_top_left_y = np.zeros(len(args))
    b_bottom_right_x  = np.zeros(len(args))
    b_bottom_right_y = np.zeros(len(args))
    
    b_bottom_right_x[0] = image1.shape[1]
    b_bottom_right_y[0] = image1.shape[0]
    
    for i in range(1,len(args)):
    #modified
        args[i] = plt.imread(args[i])
        args[i] = rgb2gray(args[i])

        # keypoints: Nx2 array of (x, y) pixel coordinates for detected features
        keypoints1 = extractor(image1)
        keypoints2 = extractor(args[i])

        # create the feature matcher and perform matching
        feature_matcher = FeatureMatcher(args)
        matches = feature_matcher(image1, keypoints1, args[i], keypoints2)

        # compute homography and get inlier matching
        data = np.column_stack((keypoints1[matches[:,0]], keypoints2[matches[:,1]]))
        H, inlier_mask = ransac(data, 100, 0.99, 10000)
        
        #return H, inlier_mask, image1, image2#xixi
        
        #for every image, do inverse homography on the corners
        from numpy.linalg import inv
        corners_T = np.array([[0,0,1],[args[i].shape[1]-1,0,1], [0,args[i].shape[0]-1,1], [args[i].shape[1]-1,args[i].shape[0]-1,1]]).T
        b_corner_coor = np.dot(inv(H), corners_T)
        b_corner_coor[:2,:] = b_corner_coor[:2,:] / b_corner_coor[2, :]
        b_corner_coor[2, :] = 1
        
        #every image2's corners on image1
        b_top_left_x[i] = np.min(b_corner_coor[0, :])      #in python, (0,0) is top left
        b_top_left_y[i] = np.min(b_corner_coor[1, :])
        b_bottom_right_x[i] = np.max(b_corner_coor[0, :])
        b_bottom_right_y[i] = np.max(b_corner_coor[1, :])
        
        
        # display the matches between the images
        plt.figure()
        plot_matches(image1, keypoints1, args[i], keypoints2, matches)

        # display the matches after RANSAC
        plt.figure()
        plot_matches(image1, keypoints1, args[i], keypoints2, matches[inlier_mask])
        
        ##########################################################################
            
            
    #create the new canvas of size that can hold all the image 
    new_width = np.max(b_bottom_right_x) - np.min(b_top_left_x) + 1
    new_height = np.max(b_bottom_right_y)  - np.min(b_top_left_y) + 1
    
    #initialize canvas
    new_canvas_shape = (int(math.ceil(new_height)), int(math.ceil(new_width)))
    canvas = np.zeros(new_canvas_shape)


    offset_x = -1 * int(np.round(np.min(b_top_left_x)))
    offset_y = -1 * int(np.round(np.min(b_top_left_y)))
    canvas[offset_y:image1.shape[0]+offset_y, offset_x:image1.shape[1]+offset_x] = image1    
                                                  
    for i in range(1,len(args)):
        #figure out the area in canvas that may include image 2
                                                                                                                                 
        b_top_left_x_canvas = int(np.ceil(b_top_left_x[i] + offset_x))
        b_top_left_y_canvas = int(np.ceil(b_top_left_y[i] + offset_y))
        b_bottom_right_x_canvas = int(np.ceil(b_bottom_right_x[i] + offset_x))
        b_bottom_right_y_canvas = int(np.ceil(b_bottom_right_y[i] + offset_y)) 
                                            
        #transform from canvas coordinate to image 2 coordinate(a is M * 3 , a.T is 3*M, r is 3*M)
        x = range(b_top_left_x_canvas, b_bottom_right_x_canvas+1)
        y = range(b_top_left_y_canvas, b_bottom_right_y_canvas+1)
        xx, yy = np.meshgrid(x,y)

        x_list  = xx.flatten()
        y_list = yy.flatten()
        a = np.column_stack((x_list, y_list))
        a = np.column_stack((a, np.ones(a.shape[0])))

        r = np.dot(H, a.T)
        r[:2,:] = r[:2,:] / r[2,:]

        #figure out which certain points that satisfy conditions of each image2 and their cooresponding points on canvas
        e = r[0, :]<=args[i].shape[1]-1
        f = 0 <=r[0, :]
        g = r[1, :]<=args[i].shape[0]-1
        h = 0<=r[1, :]
        elligible_coor = r[:, e & f & g & h]
        coordinates_in_image1 = a.T[:, e & f & g & h]

        #substitute those points on canvas with score of image2. If those points can't land on pixel of image2, use bilinear interpolation
        for idx in range(coordinates_in_image1.shape[1]):
               canvas[int(coordinates_in_image1[1,idx]), int(coordinates_in_image1[0,idx])] = bilinear_interpolation(elligible_coor[1,idx], elligible_coor[0,idx], args[i])
        


            
                                                  
                                                  
    #canvas = stitch(canvas, image2, H)



    
    plt.figure()
    plt.imshow(canvas)
    plt.show()
    ''''
    cv2.namedWindow('canvas',cv2.WINDOW_NORMAL)
    cv2.imshow("canvas", canvas)
    cv2.waitKey()'''
    
    '''
    # display the keypoints
    plt.figure(1)
    plot_keypoints(image1, keypoints1)

    plt.figure(2)
    plot_keypoints(image2, keypoints2)

    # display the matches between the images
    plt.figure(3)
    plot_matches(image1, keypoints1, image2, keypoints2, matches)

    # display the matches after RANSAC
    plt.figure(4)
    plot_matches(image1, keypoints1, image2, keypoints2, matches[inlier_mask])

    plt.show()
    '''



main(["/Users/djogem/Downloads/COMP 590-153/HW4/images/left_img.jpg", "/Users/djogem/Downloads/COMP 590-153/HW4/images/right_img.jpg"])
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and display Harris Corner features for two "
            "images, and match the features between the images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("image list", type=str, help="input image list")
    #parser.add_argument("image1", type=str, help="first input image")
    #parser.add_argument("image2", type=str, help="second input image")

    # Harris corner detection options
    parser.add_argument("--harris_corner_k", type=float, default=0.05,
        help="k-value for Harris' corner response score")
    parser.add_argument("--gaussian_sigma", type=float, default=1.,
        help="width of the Gaussian used when computing the corner response; "
             "usually set to a value between 1 and 4")
    parser.add_argument("--maxfilter_window_size", type=int, default=11,
        help="size of the (square) maximum filter to use when finding corner "
             "maxima")
    parser.add_argument("--max_num_features", type=int, default=1000,
        help="(optional) maximum number of features to extract")
    parser.add_argument("--response_threshold", type=float, default=0.01,
        help="when extracting feature points, discard any points whose corner "
             "response is less than this value times the maximum response")

    # feature description and matching options
    parser.add_argument("--matching_method", type=str, default="ssd",
        choices=set(("ssd", "ncc")),
        help="descriptor distance metric to use")
    parser.add_argument("--matching_window_size", type=int, default=7,
        help="window size (width and height) to use when matching; must be an "
             "odd number")

    parser.add_argument("--inlier_threshold", type=float, default=100.,
        help="point-to-line distance threshold, in pixels, to use for RANSAC")

    parser.add_argument("--confidence_threshold", type=float, default=0.99,
        help="stop RANSAC when the probability that a correct model has been "
             "found reaches this threshold")

    parser.add_argument("--max_num_trials", type=float, default=50000,
        help="maximum number of RANSAC iterations to allow")

    args = parser.parse_args()

    main(args)
