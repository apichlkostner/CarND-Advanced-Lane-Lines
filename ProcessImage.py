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
from GradientHelpers import abs_sobel_thresh, mag_thresh, dir_threshold, color_segmentation, \
                            mask_region_of_interest, img2gray
from LaneFit import LaneFit
from moviepy.editor import VideoFileClip
from time import time
import logging

class ProcessImage():
    def __init__(self):
        self.shape = None
        self.roi = None
        self.calCam = None
        self.laneFit = None
        self.image_cnt = 0
        self.DEBUG_IMAGE = False
        self.DEBUG_IMAGE_FOLDER = 'project_video'
        self.DEBUG_VIDEO = False
        self.segmentation = 1

    def fit(self, shape, M, MInverse, roi=None, calCam=None):
        self.shape = shape

        # region of interest
        if roi != None:
            self.roi = roi
        else:
            self.roi = { 'bottom_y':       shape[0],
                        'bottom_x_left':  int(shape[1]*0.05),
                        'bottom_x_right': int(shape[1]*0.95),
                        'top_y':          int(shape[0]*0.6),
                        'top_x_left':     int(shape[1]*0.45),
                        'top_x_right':    int(shape[1]*0.55),            
                        }
            self.roi.update({'top_center': int((self.roi['top_x_left'] + self.roi['top_x_right']) / 2)})

        if calCam != None:
            self.calCam = calCam
        else:
            self.calCam = CalibrateCamera.load()

        self.laneFit = LaneFit()

        self.M = M
        self.MInverse = MInverse

        self.update = False

    def imageClahe(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
        l_channel = hls[:,:,1]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        l_channel = clahe.apply(l_channel)
        hls[:,:,1] = l_channel

        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def writeInfo(self, img, lc=0.0, rc=0.0, mid=0.0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        pos_str = 'Left curv:  {:06.0f}'.format(lc)
        pos_str2 = 'Right curv: {:06.0f}'.format(rc)
        pos_str3 = 'Middle: {:.2f}'.format(mid)
        cv2.putText(img, pos_str ,(10,30), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, pos_str2 ,(10,50), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, pos_str3 ,(10,70), font, 0.5,(255,255,255), 1, cv2.LINE_AA)

        return img

    def process_image(self, img_orig):
        t0 = time()
        self.image_cnt += 1

        img = self.calCam.undistort(img_orig)

        if self.DEBUG_IMAGE:
            img_savename = 'image{0:04d}.jpg'.format(self.image_cnt)            
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/undist/'+img_savename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img_undist = img
        ksize = 9

        gray = img2gray(img)

        # Apply each of the thresholding functions
        if self.segmentation == 0:
            mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(20, 255))
            dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(np.pi/4*0.9, np.pi/4*1.5))
            color_seg = color_segmentation(img)

            seg_img_raw = color_seg & mag_binary & dir_binary
        
        else:
            color_seg = color_segmentation(img)
            
            kernel_size = 5
            blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
            canny = cv2.Canny(blur_gray, 40, 80).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            canny = cv2.dilate(canny, kernel, iterations = 3)
            
            seg_img_raw = (color_seg & canny)
        
        seg_img = seg_img_raw.astype(np.uint8) * 255

        seg_img[670:,:] = 0

        # mask image
        #seg_img, vertices = mask_region_of_interest(seg_img, self.roi)
        seg_img = np.dstack((seg_img, seg_img, seg_img))

        #visualization = np.dstack((np.zeros(dir_binary.shape), (dir_binary & mag_binary), color_seg)).astype(np.uint8) * 255
        
        # warp to birds eye perspective
        seg_img_warped = cv2.warpPerspective(seg_img, self.M, (seg_img.shape[1], seg_img.shape[0]), flags=cv2.INTER_LINEAR)
        
        # thresholding for interpolated pixels
        seg_img_warped = (seg_img_warped > 100).astype(np.uint8) * 255

        if self.DEBUG_IMAGE:
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/seg_warped/'+img_savename, seg_img_warped)
        
        if self.update:
            left_fit, right_fit, lane_img, lc, rc, mid = self.laneFit.procVideoImg(seg_img_warped, margin=60, numwin=20)
        else:
            histogram = np.sum(seg_img_warped[seg_img_warped.shape[0]//2:,:-2, 0], axis=0)

            midpoint = np.int(histogram.shape[0]/2)
            
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            left_fit, right_fit, lane_img, lc, rc, mid = self.laneFit.procVideoImg(seg_img_warped, leftx_base, rightx_base, margin=60, numwin=20)

            self.update = True

        if self.DEBUG_IMAGE:
            comb = cv2.addWeighted(seg_img_warped, 0.5, lane_img, 0.8, 0)
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/combined/'+img_savename, cv2.cvtColor(comb, cv2.COLOR_BGR2RGB))

        lane_img_unwarped = cv2.warpPerspective(lane_img, self.MInverse, (lane_img.shape[1], lane_img.shape[0]), flags=cv2.INTER_LINEAR)
        
        # combine original image with detection
        img = cv2.addWeighted(img, 1, lane_img_unwarped, 0.7, 0)

        # write information to image
        img = self.writeInfo(img, lc, rc, mid)
        
        # time measurement
        t1 = time()        
        logging.info('process_image: runtime = ' + str(t1))

        if self.DEBUG_VIDEO:
            img = np.dstack((np.zeros_like(seg_img_raw), canny, color_seg)).astype(np.uint8) * 255
            img = cv2.addWeighted(img, 1, lane_img_unwarped, 0.9, 0)

        if self.DEBUG_IMAGE:
            cv2.imwrite(self.DEBUG_IMAGE_FOLDER + '/result/'+img_savename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return img

def main():
    white_output = 'processed_videos/challenge_video.mp4'

    clip1 = VideoFileClip("source_videos/challenge_video.mp4").subclip(0,5)

    target_left_x = 300
    target_right_x = 1002
    target_top_y = 0
    target_bottom_y =690
    src_points = np.float32([[283, 664], [548, 480], [736, 480],  [1019, 664]])
    dst_points = np.float32([[target_left_x, target_bottom_y], [target_left_x, target_top_y],
                                [target_right_x, target_top_y], [target_right_x, target_bottom_y]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Mi = cv2.getPerspectiveTransform(dst_points, src_points)

    ld = LaneDetect()
    ld.fit((720, 1280), M, Mi)

    white_clip = clip1.fl_image(ld.process_image) # color images

    white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
    main()