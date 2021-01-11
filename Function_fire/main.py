"""
Below mention things has been performing here
1. Structural similarity under which certain size of boxes will be ignored which should 
be provided by end user along with the similarity score

2. Enter the module you want to choose True or False, here for demo purpose Fire is activated

"""

import cv2
from VideoGet import VideoGet
from helper_functions import *
from Frame_Pre_Process import *
from color_patches_detection import *
from Generate_alert import Generate_alert
import time
from datetime import datetime
from keras.preprocessing import image
# from file_upload import *
def main():
    ## Taking the value of similarity difference from the user




    """
    Taking input for binary comparision
    """
    blur_thresh = int(input("Enter the blur value of threshold: "))
    gamma_thresh = int(input("Enter the Gamma value of threshold: "))

    """
    Taking input for fire module
    """
    fire_module_status = input("Enter True for activating fire moudle : ")

    pixel_threshold_color_patch = int(input("Enter the pixel threshold for color patch for fire 500 is considered : "))
    
    # colors_hsv is should be also taken from end user.
    colors_hsv = [175, 18, 237]


    ##  Fetching it with the help of video getter
    ## RTSP link
    # video_file = "rtsp://admin:123456@202/.179.72.236:8091/H264?ch=1&subtype=0"

    # cap=cv2.VideoCapture('/home/cerium/Downloads/Function_fire/video/videoplayback.mp4')
    cap=cv2.VideoCapture(0)
    # video_file = "/home/cerium/Downloads/Structure Fire With Clear Footage of Entire Attic Lit Off!  02_26_2017.mp4"
    # video_getter = VideoGet(video_file).start()
    
    ## Here time upto which frame need to observe as well as time upto which code need not to
    ## generate alert should be given by user
    # time_in_threshold = int(input("Enter the time for which we need to observe the frame : "))
    # time_out_threshold = int(input("Enter the time for which we need not to generate any alarm : "))
    
    ##  Initializing the reference image variable with none
    prev_frame = None
    ignored_box_area = int(input("Enter the box size which need to be ignore : "))
    ## Calling generate alert class, so that it will be used for each condition
    # generate_alert = Generate_alert(time_in_threshold,time_out_threshold )

    #Load the saved model
    model = tf.keras.models.load_model('InceptionV3.h5')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Funtion for writing the image
    Processed_video = cv2.VideoWriter('case7.mp4', fourcc, 30, (720, 480))


    while True:
        ## For stopping the video getter
        # if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
        #     video_getter.stop()
        #     break
        # # Fetching the frame
        # frame = video_getter.frame
        ret, frame = cap.read()

        ## Resizing the frame
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
        curr_frame = frame.copy()
        ## For Fire alert
        if fire_module_status == "True":
            """
            MOtion detection
            """
            # # checking reference frame is available of not, if not available than 
            # #extracting it.
            gray, boxes = Motion_detection(curr_frame, prev_frame, ignored_box_area)
            prev_frame = gray
            # print("boxes:",len(boxes))
       
            # Colour patching alerts
            # """
            ## For Fire alert
            #     # print("Status", status)
            if len(boxes)>0:
                # boxes = difference_countour_boxes(diff_image, area_threshold)
                Color_patch_output = Color_patches(boxes, curr_frame,
                            pixel_threshold_color_patch, colors_hsv, blur_thresh, gamma_thresh)
                # generate_alert.main(Color_patches_status,curr_image)
                # print("Dict:",Color_patch_output)
                for i in Color_patch_output:
                    if not i:
                        # print("inside if : ", i)
                        pass
                    else:
                        # print("inside else : ", i)
                        status=model_loading(curr_frame, model)
                        print("status", status)
                        if status:
                            cv2.rectangle(curr_frame, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 3)
                            # print(i[0], i[1])
                            image = cv2.putText(curr_frame, 'Fire Detected', (i[0],i[1]- 10), cv2.FONT_HERSHEY_SIMPLEX ,1, (0, 0, 255),2, cv2.LINE_AA) 

            else:
                pass
                # print("status1", status)



        # Showing video frame.
        cv2.imshow("Main Frame", curr_frame)
        b = cv2.resize(curr_frame, (720, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        # # Writing the output
        Processed_video.write(b)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

