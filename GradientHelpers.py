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