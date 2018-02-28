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
from GradientHelpers import abs_sobel_thresh, mag_thresh, dir_threshold, color_segmentation, mask_region_of_interest, img2gray
from LaneFit import LaneFit
from moviepy.editor import VideoFileClip
from time import time
from LaneDetect import LaneDetect

if False:
    calCam = CalibrateCamera.load()
    img = mpimg.imread('test_images/straight_lines1.jpg')
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
    Mi = cv2.getPerspectiveTransform(dst_points, src_points)

    laneFit = LaneFit()
    call = 0

    def process_image(img_orig):
        global call

        t0 = time()

        img = calCam.undistort(img_orig)   

        ksize = 9

        gray = img2gray(img)

        if False:
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
            l_channel = hls[:,:,1]
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            l_channel = clahe.apply(l_channel)
            hls[:,:,1] = l_channel

            return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

        # Apply each of the thresholding functions
        mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(20, 255))
        dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(np.pi/4*0.9, np.pi/4*1.5))
        
        color_seg = color_segmentation(img, l_thresh=[30, 255], s_thresh=[160, 255])
        #print(mag_binary.shape)
        #print(dir_binary.shape)
        #print(color_seg.shape)
        #seg_img = (color_seg | (dir_binary & mag_binary)).astype(np.uint8) * 255
        seg_img_raw = (color_seg & dir_binary & mag_binary)

        seg_img = seg_img_raw.astype(np.uint8) * 255

        # mask image
        seg_img, vertices = mask_region_of_interest(seg_img, roi)

        seg_img = np.dstack((seg_img, seg_img, seg_img))

        #visualization = np.dstack((np.zeros(dir_binary.shape), (dir_binary & mag_binary), color_seg)).astype(np.uint8) * 255
        
        seg_img_warped = cv2.warpPerspective(seg_img, M, (seg_img.shape[1], seg_img.shape[0]), flags=cv2.INTER_LINEAR)

        if call == 0:
            histogram = np.sum(seg_img_warped[seg_img_warped.shape[0]//2:,:-2, 0], axis=0)

            midpoint = np.int(histogram.shape[0]/2)
            
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            left_fit, right_fit, lane_img, lc, rc = laneFit.fitLanes(seg_img_warped, leftx_base, rightx_base, margin=60, numwin=20)
        else:
            histogram = None
            midpoint = None
            leftx_base = None
            rightx_base = None
            left_fit, right_fit, lane_img, lc, rc = laneFit.fitLanes(seg_img_warped, leftx_base, rightx_base, margin=100, numwin=20)
        #call = 1


        lane_img_unwarped = cv2.warpPerspective(lane_img, Mi, (lane_img.shape[1], lane_img.shape[0]), flags=cv2.INTER_LINEAR)
        
        mask = (lane_img_unwarped[:,:,0] != 0) | (lane_img_unwarped[:,:,1] != 0) | (lane_img_unwarped[:,:,2] != 0)
        img[mask] = 0 #lane_img_unwarped[mask]

        img = img + lane_img_unwarped# np.maximum(lane_img_unwarped, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        pos_str = 'Left curv:  {:06.0f}'.format(lc)
        pos_str2 = 'Right curv: {:06.0f}'.format(rc)
        #cv2.putText(img, pos_str ,(10,30), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
        #cv2.putText(img, pos_str2 ,(10,50), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
        
        t1 = time()
        #print(t1 - t0)

        img = np.dstack((np.zeros_like(seg_img_raw), (dir_binary & mag_binary), color_seg)).astype(np.uint8) * 255 #+ 0.4 * lane_img_unwarped

        return img

    white_output = 'processed_videos/challenge_video.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("source_videos/challenge_video.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

def main():
    #white_output = 'processed_videos/challenge_video.mp4'
    #clip1 = VideoFileClip("source_videos/challenge_video.mp4")#.subclip(0,5)

    white_output = 'processed_videos/harder_challenge_video.mp4'
    clip1 = VideoFileClip("source_videos/harder_challenge_video.mp4")#.subclip(0,5)

    original_bottom_left_x = 283
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
