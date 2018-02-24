import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as grd
import pickle
import time
import os.path
import glob
from CameraCalibration import CalibrateCamera
from GradientHelpers import abs_sobel_thresh, mag_thresh, dir_threshold, color_segmentation, mask_region_of_interest
from LaneFit import LaneFit
    
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
    img_orig = mpimg.imread('test_images/straight_lines1.jpg')
    img_orig = mpimg.imread('test_images/test6.jpg')
    #img_orig = mpimg.imread('camera_cal/calibration1.jpg')

    img = calCam.undistort(img_orig)

    # define region of interest
    roi = { 'bottom_y':       img.shape[0],
            'bottom_x_left':  int(img.shape[1]*0.05),
            'bottom_x_right': int(img.shape[1]*0.95),
            'top_y':          int(img.shape[0]*0.6),
            'top_x_left':     int(img.shape[1]*0.45),
            'top_x_right':    int(img.shape[1]*0.55),            
          }
    roi.update({'top_center': int((roi['top_x_left'] + roi['top_x_right']) / 2)})
    
    horizon = 425

    original_bottom_left_x = 283
    target_left_x = 300
    target_right_x = 1002
    target_top_y = 0
    target_bottom_y =690
    src_points = np.float32([[283, 664], [548, 480], [736, 480],  [1019, 664]])
    dst_points = np.float32([[target_left_x, target_bottom_y], [target_left_x, target_top_y],
                             [target_right_x, target_top_y], [target_right_x, target_bottom_y]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    ksize = 9 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 255))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(60, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(np.pi/4*0.9, np.pi/4*1.5))
    
    color_seg = color_segmentation(img, s_thresh=[160, 255])
    
    seg_img = (color_seg | (dir_binary & mag_binary)).astype(np.uint8) * 255

    # mask image
    seg_img, vertices = mask_region_of_interest(seg_img, roi)

    seg_img = np.dstack((seg_img, seg_img, seg_img))

    visualization = np.dstack((np.zeros(dir_binary.shape), (dir_binary & mag_binary), color_seg)).astype(np.uint8) * 255
    
    seg_img_warped = cv2.warpPerspective(seg_img, M, (seg_img.shape[1], seg_img.shape[0]), flags=cv2.INTER_LINEAR)
    print(seg_img_warped.shape)
    histogram = np.sum(seg_img_warped[gradx.shape[0]//2:,:-2, 0], axis=0)

    out_img = seg_img_warped.copy()
    midpoint = np.int(histogram.shape[0]/2)
    
    print('Midpoint = ' + str(midpoint))
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    print('Bases = ' + str((leftx_base, rightx_base)))

    laneFit = LaneFit()
    left_fit, right_fit, lane_img = laneFit.fitLanes(seg_img_warped, leftx_base, rightx_base, margin=100)


    # Plot the result    
    gs = grd.GridSpec(3, 2, height_ratios=[10,10,10], width_ratios=[1,1], wspace=0.1)

    ax = plt.subplot(gs[0,0])            
    ax.imshow(img_orig)
    ax.set_title('Original Image', fontsize=10)

    ax = plt.subplot(gs[0,1])
    ax.imshow(img)
    ax.set_title('Undistorted Image', fontsize=10)

    ax = plt.subplot(gs[1,0])
    ploty = np.linspace(0, lane_img.shape[0]-1, lane_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ax.imshow(lane_img)
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    
    ax.set_title('Lane fit', fontsize=10)

    ax = plt.subplot(gs[1,1])
    ax.imshow(seg_img_warped, cmap='gray')
    ax.set_title('Combined', fontsize=10)

    ax = plt.subplot(gs[2,0])
    ax.imshow(visualization)
    ax.set_title('Visualization', fontsize=10)

    ax = plt.subplot(gs[2,1])
    ax.plot(histogram)
    ax.set_title('Histogram', fontsize=10)

    plt.show()


if __name__ == "__main__":
    main()