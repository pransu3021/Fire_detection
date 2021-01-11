"""
Here we will find the color pigment is avaiable or not. Below mention function is implemented here
1. Color_patches : return patches detected or not, along with the other necessary details.
2. pixel_count_roi : It is helping function for color patches, which return pixel count and roi.
3. get_roi_frame : It is also helping function, which will give the boxes in which pigment is detected.

"""

import cv2
from Frame_Pre_Process import *
import numpy as np

def Color_patches(boxes, curr_image, pixel_threshold_color_patch, colors_hsv, blur_thresh, gamma_thresh):
    detected_boxes = []
    Color_patch_output = {} 
    for i in boxes:
        xa1, ya1, xa2, ya2 = i[:4]
        img_iou = curr_image[int(ya1):int(ya2), int(xa1):int(xa2)]
        # cv2.imshow("img_iou", img_iou)
        

        pixel_count, roi = pixel_count_roi(img_iou,colors_hsv, blur_thresh , gamma_thresh)
        # if pixel_count > 0:
        #     print("pixel_count", pixel_count)
        if int(pixel_count) > pixel_threshold_color_patch:

            if len(roi) > 0:
                xg1, yg1, xg2, yg2 = get_roi_frame(roi)

                xb1 = xg1 + xa1
                yb1 = yg1 + ya1
                xb2 = xb1 + (xg2-xg1)
                yb2 = yb1 + (yg2-yg1)
                detected_boxes.append([xb1, yb1, xb2, yb2])
                # Adding a new key value pair
                list = tuple([xb1, yb1, xb2, yb2])
                Color_patch_output.update({list : True})                
                # return True, detected_boxes
            else:
                Color_patch_output.update({False: False})  
        else:
            Color_patch_output.update({False: False})  
    return Color_patch_output


def pixel_count_roi(frame,colors_hsv, blur_thresh, gamma_thresh):

    """
    :param frame:Frame from main is input
    :param lower:
    :param upper:
    :param blur_thresh:
    :param gamma_thresh:
    :return:
    """
    # blur_thresh = 100
    # gamma_thresh = 100
    # Formula for calculating upper and lower limit
    upper = np.array([colors_hsv[0] + 10, colors_hsv[1] + 10, colors_hsv[2] + 40])
    lower = np.array([colors_hsv[0] - 10, colors_hsv[1] - 10, colors_hsv[2] - 40])
    if blur_thresh >0:
        # print("inside if")
        preprocess = PreProcess(blur_thresh, gamma_thresh)
        frame = preprocess.gamma_correction(frame)

        # blur = cv2.GaussianBlur(frame, (9, 9), 0)
        # print("hsv if: ", len(blur))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)

        output = cv2.bitwise_and(frame, hsv, mask=mask)

        no_red = cv2.countNonZero(mask)
        roi = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        return no_red, roi
    else:
        # blur = cv2.GaussianBlur(frame, (9, 9), 0)
        # print("hsv else: ", type(blur))
        # if blur.size > 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imwrite("hsv.jpg", hsv)
        mask = cv2.inRange(hsv, lower, upper)

        output = cv2.bitwise_and(frame, hsv, mask=mask)
        no_red = cv2.countNonZero(mask)
        roi = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        return no_red, roi

def get_roi_frame(roi):
    """
    Draw bounding box
    :param roi: input from the throat function
    :return: return the rectangle box
    """
    boxes = []
    for c in roi:
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])
    boxes = np.asarray(boxes)
    x1, y1 = np.min(boxes, axis=0)[:2]
    w, h = np.max(boxes, axis=0)[2:]
    return x1, y1,  x1+w, y1+h

