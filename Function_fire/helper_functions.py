"""
The following function has been implemented in this script
1. Structural Similarity : It will return the image is similar or not
2. difference_countour_boxes : It will return the bounding boxes of the difference of the image.
3. bb_intersection_over_union : It will return the overlap of two object.
4. Binary_compare : It will return whether illumination detected or not, for eg
    a. Flash detected
    b. spark detected
    c. light source detected
"""
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image


from skimage.measure import compare_ssim
import cv2
import imutils
import numpy as np
import json
import time
import datetime
# import requests

def Motion_detection(curr_frame, prev_frame, ignored_box_area):
    gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    # Converting gray scale image to GaussianBlur 
	# so that change can be find easily 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 
    boxes = []
    if prev_frame is not None:
        diff_frame = gray - prev_frame
        diff_frame -= diff_frame.min()
        disp_frame = np.uint8(255.0*diff_frame/float(diff_frame.max()))
        thresh_frame = cv2.threshold(disp_frame, 30, 255, cv2.THRESH_BINARY)[1] 
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
        cnts,_ = cv2.findContours(thresh_frame.copy(), 
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        # print("cnts:",len(cnts))
        

        for contour in cnts: 
            (x, y, w, h) = cv2.boundingRect(contour)
            if w*h > ignored_box_area:
                boxes.append([x, y, x+w, y+h])
                # making green rectangle arround the moving object 
                # cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return gray, boxes



def bb_intersection_over_union(boxes1, boxes2, intersection_percent):

    iou_boxes= []
    # print("person_bboxes", person_bboxes)
    # print("boxes", boxes)
    for boxA in boxes1:

        for boxB in boxes2:
            # print("boxB", boxB)
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # compute the area of intersection rectangle
            interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
            if interArea == 0:
                # a =  0
                iou_boxes.append(boxB)
                # iou.append(a)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
            boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            a = interArea / float(boxAArea + boxBArea - interArea)
            if a < intersection_percent:
                iou_boxes.append(boxB)

    return iou_boxes

def model_loading(frame, model):
        im = Image.fromarray(frame, 'RGB')
        #Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224,224))
        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        #if prediction is 0, which means there is fire in the frame.
        # print("prediction:", prediction)
        if prediction == 0:
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                if probabilities[prediction] > 0.80:
                        # print(probabilities[prediction])
                        return True
                else:
                        return False
        else:
                return False