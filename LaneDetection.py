import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import time
import os.path
import glob
from CameraCalibration import CalibrateCamera
from GradientHelpers import abs_sobel_thresh, mag_thresh, dir_threshold

    
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
    #img_orig = mpimg.imread('test_images/straight_lines2.jpg')
    img_orig = mpimg.imread('test_images/test4.jpg')
    #img_orig = mpimg.imread('camera_cal/calibration1.jpg')

    img = calCam.undistort(img_orig)

    horizon = 425

    #src_points = np.float32([[283, 664], [616, 438], [664, 438],  [1019, 664]])
    #dst_points = np.float32([[283, 664], [283, 0], [1019, 0], [1019, 664]])
    src_points = np.float32([[283, 664], [542, 480], [732, 480],  [1019, 664]])
    dst_points = np.float32([[283, 719], [283, 200], [1019, 200], [1019, 719]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    if False:
        # Choose a Sobel kernel size
        ksize = 15 # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 255))
        grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 255))
        mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(50, 255))
        dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))


        # Plot the result
        f, ax = plt.subplots(3, 2, figsize=(24, 9))

        f.tight_layout()

        ax[0, 0].imshow(img_orig)
        ax[0, 0].set_title('Original Image', fontsize=10)

        ax[0, 1].imshow(img)
        ax[0, 1].set_title('Undistorted Image', fontsize=10)

        ax[1, 0].imshow(gradx, cmap='gray')
        ax[1, 0].set_title('Grad_x', fontsize=10)

        ax[1, 1].imshow(grady, cmap='gray')
        ax[1, 1].set_title('Grad_y', fontsize=10)

        ax[2, 0].imshow(dir_binary, cmap='gray')
        ax[2, 0].set_title('Direction', fontsize=10)

        ax[2, 1].imshow(dir_binary & mag_binary & gradx, cmap='gray')
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

        ax[1, 0].imshow(gradx, cmap='gray')
        ax[1, 0].set_title('Grad_x', fontsize=5)

        ax[1, 1].plot(histogram)
        ax[1, 1].set_title('Histogram', fontsize=5)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()


if __name__ == "__main__":
    main()