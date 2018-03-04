import numpy as np
import cv2
from CameraCalibration import CalibrateCamera
from LaneFit import LaneFit
from ProcessImage import ProcessImage
from moviepy.editor import VideoFileClip

def main():
    white_output = 'processed_videos/challenge_video.mp4'
    clip1 = VideoFileClip("source_videos/challenge_video.mp4")#.subclip(0,5)

    #white_output = 'processed_videos/harder_challenge_video.mp4'
    #clip1 = VideoFileClip("source_videos/harder_challenge_video.mp4")#.subclip(0,5)

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

    ld = ProcessImage()
    ld.fit((720, 1280), M, Mi)

    white_clip = clip1.fl_image(ld.process_image) # color images

    white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
    main()
