import numpy as np
import cv2

class LaneFit():
    def __init__(self):
        img = None
        self.left_fit = None
        self.right_fit = None
        self.firstCall = True
        self.fircal = True
        self.fit = False
        self.middle = 0.0

    def update(self, img):
        if not self.firstCall:
            left_lane  = self.left_fit[0]*(nonzero_y**2) + self.left_fit[1]*nonzero_y + self.left_fit[2]
            right_lane = self.right_fit[0] * (nonzero_y**2) + self.right_fit[1] * nonzero_y + self.right_fit[2]

            left_lane_inds =  ((nonzero_x > left_lane - margin)) & (nonzero_x < left_lane + margin)
            right_lane_inds = ((nonzero_x > right_lane - margin)) & (nonzero_x < right_lane + margin)

            if False:
                if len(left_lane_inds) > 1:
                    window_mean = np.int(np.mean(nonzero_x[left_lane_inds]))
                    # remove outliers
                    mask = np.abs(nonzero_x - window_mean)
                    left_lane_inds[mask > 20] = False

                    if left_lane_inds.sum() > 0:
                        window_mean = np.int(np.mean(nonzero_x[left_lane_inds]))

                if len(right_lane_inds) > 1:
                    window_mean = np.int(np.mean(nonzero_x[right_lane_inds]))
                    # remove outliers
                    mask = np.abs(nonzero_x - window_mean)
                    right_lane_inds[mask > 20] = False

                    if left_lane_inds.sum() > 0:
                        window_mean = np.int(np.mean(nonzero_x[right_lane_inds]))

            # Again, extract left and right line pixel positions
            leftx = nonzero_x[left_lane_inds]
            lefty = nonzero_y[left_lane_inds] 
            rightx = nonzero_x[right_lane_inds]
            righty = nonzero_y[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        
    def fitCurves(self, left_lane_inds, right_lane_inds, nonzero_y, nonzero_x, kr=9.):
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds] 
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds] 

        # Fit a second order polynomial to each
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        if len(lefty) > 40 and len(righty) > 40:
            self.fit = True
            if self.fircal:
                self.left_fit = np.polyfit(lefty, leftx, 2)
                self.right_fit = np.polyfit(righty, rightx, 2)
                self.left_fit_phys = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
                self.right_fit_phys = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
                self.fircal = False
            else:
                poly_l = np.polyfit(lefty, leftx, 2)
                poly_r = np.polyfit(righty, rightx, 2)
                self.left_fit = (kr * self.left_fit + poly_l) / (kr + 1.)
                self.right_fit = (kr * self.right_fit + poly_r) / (kr + 1.)
                poly_l = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
                poly_r = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
                self.left_fit_phys = (kr * self.left_fit_phys + poly_l) / (kr + 1.)
                self.right_fit_phys = (kr * self.right_fit_phys + poly_r) / (kr + 1.)
            
            # middle point
            left_fitx = self.left_fit[2]
            right_fitx = self.right_fit[2]            
            self.middle = ((self.shape[1] / 2) - ((left_fitx + right_fitx) / 2)) * xm_per_pix

    def plotCurves(self, img):
        margin = 15
            
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        # curve
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        inbetween = np.hstack((left_line_window2, right_line_window1))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(img, np.int_([left_line_pts]), (255,0, 0))
        cv2.fillPoly(img, np.int_([inbetween]), (180,180, 0))
        cv2.fillPoly(img, np.int_([right_line_pts]), (0,255, 0))

        return img

    def fitLanes(self, img, leftx_base=None, rightx_base=None, numwin=9, margin=100, minpix=50):
        self.shape = img.shape
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

            # recenter next window on their mean position
            if len(good_left_inds) > minpix:
                window_mean = np.int(np.mean(nonzero_x[good_left_inds]))
                # remove outliers
                good_left_inds = good_left_inds[np.abs(nonzero_x[good_left_inds] - window_mean) < 20]

                if len(good_left_inds) > 1:
                    window_mean = np.int(np.mean(nonzero_x[good_left_inds]))
                    leftx_current = window_mean
            if len(good_right_inds) > minpix:
                window_mean = np.int(np.mean(nonzero_x[good_right_inds]))
                # remove outliers
                good_right_inds = good_right_inds[np.abs(nonzero_x[good_right_inds] - window_mean) < 20]

                if len(good_right_inds) > 1:
                    window_mean = np.int(np.mean(nonzero_x[good_right_inds]))
                    rightx_current = window_mean

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        self.fitCurves(left_lane_inds, right_lane_inds, nonzero_y, nonzero_x)

        window_img = np.zeros_like(out_img)
        if self.fit:
            # plot lane borders and area
            window_img  = self.plotCurves(window_img)

            y_eval = window_img.shape[0] - 1

            left_curverad = ((1 + (2*self.left_fit_phys[0]*y_eval + self.left_fit_phys[1])**2)**1.5) \
                                / np.absolute(2*self.left_fit_phys[0])
            right_curverad = ((1 + (2*self.right_fit_phys[0]*y_eval + self.right_fit_phys[1])**2)**1.5) \
                                / np.absolute(2*self.right_fit_phys[0])
        else:
            left_curverad = 1
            right_curverad = 1
        
        #out_img = img.copy()
        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        return self.left_fit, self.right_fit, window_img, left_curverad, right_curverad, self.middle