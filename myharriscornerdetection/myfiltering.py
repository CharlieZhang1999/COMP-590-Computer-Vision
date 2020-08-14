r"""
COMP590 Assignment2
Image filtering
Qiuyang Zhang
"""
import numpy as np
import math
import cv2 


def cross_correlation_2d(im, kernel, path='same', padding='zero'):
    '''
    Inputs:
        im: input image (RGB or grayscale)
        kernel: input kernel
        path: 'same', 'valid', 'full' filtering path
		padding: 'zero', 'replicate'
    Output:
        filtered image
    '''
    #Assuming the input kernel is square kernel
    #determin padding at  row and column
    k_size = kernel.shape[0]
    #half-width
    k_step = math.floor(k_size/2)
   
    # Fill in
    if(path=='valid'):
        if len(im.shape)>2:
            output = np.zeros([im.shape[0]-2*k_step,im.shape[1]-2*k_step,im.shape[2]])#,dtype=np.uint8)
            padding_img = im
            
            for i in range (0,im.shape[0]-2*k_step):
                for j in range(0,im.shape[1]-2*k_step):
                    for n in range(0,im.shape[2]):
                        output[i,j,n]= np.sum(padding_img[i+k_step-k_step:i+k_step+(k_step+1),j+k_step-k_step:j+k_step+(k_step+1),n]*kernel)
                        output = output.astype('int')
    
        else:
            output = np.zeros([im.shape[0]-2*k_step,im.shape[1]-2*k_step])#,dtype=np.uint8)
            padding_img = im
            
            for i in range (0,im.shape[0]-2*k_step):
                for j in range(0,im.shape[1]-2*k_step):
                    output[i,j]= np.sum(padding_img[i+k_step-k_step:i+k_step+(k_step+1),j+k_step-k_step:j+k_step+(k_step+1)]*kernel)
                    output = output.astype('int')
            
    
    elif(path=='same'):
    
        if len(im.shape)>2:
            output = np.zeros([im.shape[0],im.shape[1],im.shape[2]])#,dtype=np.uint8)
            
            #create a padding image
            padding_img = np.zeros([im.shape[0]+2*k_step, im.shape[1]+2*k_step, im.shape[2]]).astype('int')
            padding_img[k_step:-k_step,k_step:-k_step,:] = im

            for i in range (0,im.shape[0]):
                for j in range(0,im.shape[1]):
                    for n in range(0,im.shape[2]):
                        output[i,j,n]= np.sum(padding_img[i+k_step-k_step:i+k_step+(k_step+1),j+k_step-k_step:j+k_step+(k_step+1),n]*kernel)
                        output = output.astype('int')
        else:
            output = np.zeros([im.shape[0],im.shape[1]])#,dtype=np.uint8)
            padding_img = np.zeros([im.shape[0]+2*k_step, im.shape[1]+2*k_step]).astype('int')
            padding_img[k_step:-k_step,k_step:-k_step] = im
    
            for i in range (0,im.shape[0]):
                for j in range(0,im.shape[1]):
                    output[i,j]= np.sum(padding_img[i+k_step-k_step:i+k_step+(k_step+1),j+k_step-k_step:j+k_step+(k_step+1)]*kernel)
                    output = output.astype('int')
        
    elif(path=='full'):
        if len(im.shape)>2:
            output = np.zeros([im.shape[0]+2*k_step, im.shape[1]+2*k_step, im.shape[2]])#,dtype=np.uint8)
            
            #create a padding image
            padding_img = np.zeros([im.shape[0]+4*k_step, im.shape[1]+4*k_step, im.shape[2]]).astype('int')
            padding_img[2*k_step:-2*k_step,2*k_step:-2*k_step,:] = im

            for i in range (0,im.shape[0]+2*k_step):
                for j in range(0,im.shape[1]+2*k_step):
                    for n in range(0,im.shape[2]):
                        output[i,j,n]= np.sum(padding_img[i+k_step-k_step:i+k_step+(k_step+1),j+k_step-k_step:j+k_step+(k_step+1),n]*kernel)
                        output = output.astype('int')
        else:
            output = np.zeros([im.shape[0]+2*k_step, im.shape[1]+2*k_step])
            padding_img = np.zeros([im.shape[0]+4*k_step, im.shape[1]+4*k_step]).astype('int')
            padding_img[2*k_step:-2*k_step,2*k_step:-2*k_step] = im
            
            for i in range (0,im.shape[0]+2*k_step):
                for j in range(0,im.shape[1]+2*k_step):
                    output[i,j]= np.sum(padding_img[i+k_step-k_step:i+k_step+(k_step+1),j+k_step-k_step:j+k_step+(k_step+1)]*kernel)
                    output = output.astype('int')
    output = output.astype(np.float32)
    return output
    
    

def convolve_2d(im, kernel, path='same', padding='zero'): 
    '''
    Inputs:
        im: input image (RGB or grayscale)
        kernel: input kernel
        path: 'same', 'valid', 'full' filtering path
		padding: 'zero', 'replicate'
    Output:
        filtered image
    '''
    
    # Flip kernel
    K = kernel[::-1,::-1]
        
        
    #filtering path
    if(path=='same'):
        output = cross_correlation_2d(im, K)
    elif(path=='valid'):
        output = cross_correlation_2d(im, K, 'valid')
    elif(path=='full'):
        output = cross_correlation_2d(im, K, 'full')
    # Call cross_correlation_2d
    
    return output


def gaussian_blur_kernel_2d(k_size, sigma):
    '''
    Inputs:
        k_size: kernel size
        sigma: standard deviation of Gaussian distribution
    Output:
        Gaussian kernel
    '''
    
    # Fill in
    m = k_size
    gaussian_k = np.ones((m,m))
    m = math.floor(m/2)
    for x in range(-m,m+1):
        for y in range(-m,m+1):
            gaussian_k[x+m,y+m] = math.exp(-(x**2+y**2)/(2*sigma**2))
     
    #normalization
    summation = np.sum(gaussian_k)
    gaussian_k = gaussian_k/summation
    return gaussian_k

def image_shrinking(im, dim):
    '''
    Inputs:
        im: input image (RGB or grayscale)
        dim: output image size 
    Output:
        Downsampled image
    '''    
    
    #if(len(dim)>2):
        #dim = (int(dim[0]),int(dim[1],int(dim[2])))
    #else:
        #dim = (int(dim[0]),int(dim[1]))
        
    # Filter the input image using Gaussian kernel
    #mark
    g_k = gaussian_blur_kernel_2d(3, 1)
    img = cross_correlation_2d(im,g_k)
    img = img.astype(np.uint8)
    # Resize the filtered image to output size dim
    new_im = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    print(new_im)
    
    return new_im    
    
def sobel_kernel():
    '''
    Output:
        Sobel kernels for x and y direction
    '''    
    sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    return sobel_x, sobel_y

def sobel_image(im):
    '''
    Inputs:
        im: input image (RGB or grayscale)
    Output:
        Gradient magnitude
        Derivative of image in x direction
        Derivative of image in y direction
        (All need to be normalized for visualization)
    '''
    
    # Convert image to grayscale if it is an RGB image
    im_c = im.copy()
    if(len(im.shape)>2):
            im_c = cv2.cvtColor(im_c,cv2.COLOR_BGR2GRAY)
            
            
    
    # Fill in
    sobel_x, sobel_y = sobel_kernel()
    
    
    #padding step
    
    
    #derivative map
    derivative_x = convolve_2d(im_c, sobel_x, path='valid', padding='zero')
    derivative_y = convolve_2d(im_c, sobel_y, path='valid', padding='zero')
    
    magnitude = (derivative_x**2+derivative_y**2)**0.5
    magnitude = magnitude.astype(np.float32)
    
    #normalization
    derivative_x = abs(derivative_x)
    derivative_y = abs(derivative_y)
    magnitude = abs(magnitude)
    
    #rescaling
    derivative_x = (derivative_x-np.max(derivative_x))/(np.max(derivative_x)-np.min(derivative_x))
    derivative_y = (derivative_y-np.max(derivative_y))/(np.max(derivative_y)-np.min(derivative_y))
    magnitude = (magnitude-np.max(magnitude))/(np.max(magnitude)-np.min(magnitude))
    
    #change into 0-255, integer. No negative value, so can use np.uint8
    derivative_x = (derivative_x*255).astype(np.uint8)
    derivative_y = (derivative_y*255).astype(np.uint8)
    magnitude = (magnitude*255).astype(np.uint8)
    
    return derivative_x, derivative_y, magnitude



