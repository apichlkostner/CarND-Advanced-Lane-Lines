import numpy as np
import cv2

class LaneFit():
    def __init__(self):
        img = None

    def fitLanes(self, img, leftx_base, rightx_base, numwin=9, margin=100, minpix=50):
        out_img = img.copy()
        window_height = np.int(img.shape[0] / numwin)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(numwin):
            # Identify window boundaries in x and y (and right and left)
            win_y_low  = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low   = leftx_current - margin
            win_xleft_high  = leftx_current + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            out_img = cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                                    (win_xleft_high, win_y_high), (0,255,0), 2)
            out_img = cv2.rectangle(out_img, (win_xright_low, win_y_low),
                                    (win_xright_high, win_y_high), (255,0,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
            (nonzero_x >= win_xleft_low) &  (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
            (nonzero_x >= win_xright_low) &  (nonzero_x < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # recenter next window on their mean position
            if len(good_left_inds) > minpix:
                window_mean = np.int(np.mean(nonzero_x[good_left_inds]))
                # remove outliers
                near_mean = good_left_inds[np.abs(nonzero_x[good_left_inds] - window_mean) < 20]
                window_mean = np.int(np.mean(nonzero_x[near_mean]))

                leftx_current = np.int((leftx_current + window_mean) / 2)
                
                #if (np.abs(leftx_current - window_mean) < 50):
                #    leftx_current - window_mean
            if len(good_right_inds) > minpix:
                window_mean = np.int(np.mean(nonzero_x[good_right_inds]))
                # remove outliers
                near_mean = good_right_inds[np.abs(nonzero_x[good_right_inds] - window_mean) < 20]
                window_mean = np.int(np.mean(nonzero_x[near_mean]))

                rightx_current = np.int((rightx_current + window_mean) / 2)
                #if (np.abs(rightx_current - window_mean) < 50):
                #    rightx_current - window_mean 

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds] 
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        return left_fit, right_fit, out_img