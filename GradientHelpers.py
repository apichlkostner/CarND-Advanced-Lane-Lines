import numpy as np
import cv2

def img2gray(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    gray_image = clahe.apply(gray_image)

    return gray_image

def abs_sobel_thresh(gray, sobel_kernel=3, orient='x', thresh=(0, 255)):
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
    sobel_abs = np.abs(sobel)

    scaled = np.uint8(255 * sobel_abs / sobel_abs.max())

    mask = (scaled > thresh[0]) & (scaled < thresh[1])
    
    return mask
    

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.sqrt(sobelx*sobelx + sobely*sobely)
    scaled = np.uint8(255 * sobel_abs / sobel_abs.max())

    mask = ((scaled > mag_thresh[0]) & (scaled < mag_thresh[1])).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    return mask
    

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):    
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    direction = np.arctan2(sobely, sobelx)
    
    mask = ((direction > thresh[0]) & (direction < thresh[1])).astype(np.uint8)

    print('Direction = ' + str(direction))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    
    return mask

def color_segmentation(img, l_thresh=[30, 255], s_thresh=[90, 255]):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) #.astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    l_channel = clahe.apply(l_channel)
    l_thresh = [170,255]
    s_thresh = [0, 60]
    # Threshold color channel
    white = (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
    white &= (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])
    yellow = (h_channel >= 16) & (h_channel <= 23) & (s_channel >= 120)
    yellow &= (l_channel >= 40) & (l_channel <= 200)
    
    s_binary = white | yellow

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
