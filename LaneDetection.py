import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import time
import os.path
import glob
from CameraCalibration import CalibrateCamera
from GradientHelpers import abs_sobel_thresh, mag_thresh, dir_threshold, color_segmentation, mask_region_of_interest

    
def main():
    calCam = CalibrateCamera.load()

    if calCam == None:
        images = glob.glob('camera_cal/calibration*.jpg')

        calCam = CalibrateCamera()

        calCam.findCorners(images, (9, 6))

        calCam.calibrateCamera()

        calCam.write()

    print(calCam.mtx)
    print(calCam.dist)
    # Read in an image
    img_orig = mpimg.imread('test_images/straight_lines2.jpg')
    #img_orig = mpimg.imread('test_images/test4.jpg')
    #img_orig = mpimg.imread('camera_cal/calibration1.jpg')

    img = calCam.undistort(img_orig)

    # define region of interest
    roi = { 'bottom_y':       img.shape[0]*0.9,
            'bottom_x_left':  int(img.shape[1]*0.05),
            'bottom_x_right': int(img.shape[1]*0.95),
            'top_y':          int(img.shape[0]*0.6),
            'top_x_left':     int(img.shape[1]*0.45),
            'top_x_right':    int(img.shape[1]*0.55),            
          }
    roi.update({'top_center': int((roi['top_x_left'] + roi['top_x_right']) / 2)})
    
    

    horizon = 425

    #src_points = np.float32([[283, 664], [616, 438], [664, 438],  [1019, 664]])
    #dst_points = np.float32([[283, 664], [283, 0], [1019, 0], [1019, 664]])
    original_bottom_left_x = 283
    target_left_x = 300
    target_right_x = 1002
    target_top_y = 0
    target_bottom_y =719
    src_points = np.float32([[283, 664], [542, 480], [732, 480],  [1019, 664]])
    dst_points = np.float32([[target_left_x, target_bottom_y], [target_left_x, target_top_y],
                             [target_right_x, target_top_y], [target_right_x, target_bottom_y]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    if True:
        print (np.arctan2(0.5, -0.5))
        print (np.arctan2(1, 0))
        print (np.arctan2(0.5, 0.5))
        # Choose a Sobel kernel size
        ksize = 9 # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 255))
        grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 255))
        mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(60, 255))
        dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(np.pi/4*0.9, np.pi/4*1.1))
        
        color_seg = color_segmentation(img, s_thresh=[160, 255])
        

        seg_img = (color_seg | (dir_binary & mag_binary)).astype(np.uint8) * 255

        # mask image
        seg_img, vertices = mask_region_of_interest(seg_img, roi)

        seg_img = np.dstack((seg_img, seg_img, seg_img))
        
        seg_img_warped = cv2.warpPerspective(seg_img, M, (seg_img.shape[1], seg_img.shape[0]), flags=cv2.INTER_LINEAR)

        # Plot the result
        f, ax = plt.subplots(3, 2, figsize=(24, 9))

        f.tight_layout()

        ax[0, 0].imshow(img_orig)
        ax[0, 0].set_title('Original Image', fontsize=10)

        ax[0, 1].imshow(img)
        ax[0, 1].set_title('Undistorted Image', fontsize=10)

        ax[1, 0].imshow(color_seg, cmap='gray')
        ax[1, 0].set_title('color_seg', fontsize=10)

        ax[1, 1].imshow(mag_binary, cmap='gray')
        ax[1, 1].set_title('mag_binary', fontsize=10)

        ax[2, 0].imshow(seg_img, cmap='gray')
        ax[2, 0].set_title('Direction', fontsize=10)

        ax[2, 1].imshow(seg_img_warped, cmap='gray')
        ax[2, 1].set_title('Combined', fontsize=10)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.show()
    else:
        gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=9, thresh=(40, 255))

        histogram = np.sum(gradx[gradx.shape[0]//2:,:-2], axis=0)

        f, ax = plt.subplots(2, 2, figsize=(24, 9))

        f.tight_layout()

        ax[0, 0].imshow(img_orig)
        ax[0, 0].set_title('Original Image', fontsize=5)

        ax[0, 1].imshow(img)
        ax[0, 1].set_title('Undistorted Image', fontsize=5)

        ax[1, 0].imshow(dir_binary, cmap='gray')
        ax[1, 0].set_title('Grad_x', fontsize=5)

        ax[1, 1].plot(histogram)
        ax[1, 1].set_title('Histogram', fontsize=5)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()


if __name__ == "__main__":
    main()