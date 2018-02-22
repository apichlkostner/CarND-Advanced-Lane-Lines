import numpy as np
import cv2


def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
    sobel_abs = np.abs(sobel)

    scaled = np.uint8(255 * sobel_abs / sobel_abs.max())

    mask = (scaled > thresh[0]) & (scaled < thresh[1])
    
    return mask
    

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.sqrt(sobelx*sobelx + sobely*sobely)
    scaled = np.uint8(255 * sobel_abs / sobel_abs.max())

    mask = (scaled > mag_thresh[0]) & (scaled < mag_thresh[1])
    
    return mask
    

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    direction = np.arctan2(sobely, sobelx)
    
    mask = (direction > thresh[0]) & (direction < thresh[1])
    
    return mask

def color_segmentation(img, s_thresh=[90, 255]):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Threshold color channel
    
    s_binary = (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
    
    return s_binary


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def mask_region_of_interest(img, roi):
    """ 
    calculates a mask and the vertices of the region of interest
    
    Parameters
    ----------
    img : np.array
        image
    roi : dictionary
        parameter dictionary

    Returns
    -------
    np.array, np.array
        mask and vertices of the mask
    """
    
    mask = np.zeros_like(img)   
    ignore_mask_color = 255   

    # generate mask for region of interest use it on 
    vertices = np.array([[(roi['bottom_x_left'], roi['bottom_y']),
                          (roi['top_x_left'], roi['top_y']),
                          (roi['top_x_right'], roi['top_y']),
                          (roi['bottom_x_right'], roi['bottom_y'])]],
                        dtype=np.int32)
                          
    cv2.fillPoly(mask, vertices, ignore_mask_color)
                          
    img_masked_edges = cv2.bitwise_and(img, mask)
    
    return img_masked_edges, vertices
