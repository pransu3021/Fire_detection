import time
from datetime import datetime
import cv2

class Generate_alert:
    def __init__(self,time_in_threshold,time_out_threshold ):
        # self.status = status
        # self.curr_image = curr_image
        self.time_flag =  False
        self.time_out_flag = False
        self.alarm_flag = False
        self.start_time = None
        self.time_out_timer = None
        self.time_in_threshold = time_in_threshold
        self.time_out_threshold = time_out_threshold

    
    def main(self, status, curr_image):
        if status:
            if not self.time_flag:
                self.start_time = time.time()
                self.time_flag = True
            else:
                time_diff = time.time() - self.start_time
                print("time_diff", time_diff)
                if time_diff > self.time_in_threshold:
                    if self.alarm_flag is False:
                        print("time_diff correct", time_diff)
                        print("**********************alarm generated*************************")
                        filename = datetime.now().strftime("%d-%m-%Y_%I_%M_%S_%p")
                        path = "./data/"
                        cv2.imwrite(path + filename + ".png", curr_image)
                        self.alarm_flag = True
                        self.time_out_timer = time.time()
                        self.time_out_flag = True
                        # upload_blob_instance = BlobUpload()
                        # blob_upload_status=upload_blob_instance.upload_file(filename + ".png")
                        # if blob_upload_status is None:
                        #     alarm_message = "Fire Detected"
                        #     image = filename + ".png"
                        #     # api_request(alarm_message, image)
                        #     alarm_flag = True
                        #     time_out_timer = time.time()
                        #     time_out_flag = True
                        # cv2.putText(curr_frame, "Fire Detected", (int(10), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (255, 0, 0), 2)

                        # else:
                        #     pass
                    else:
                        time_out_timer_diff = time.time() - self.time_out_timer

                        if time_out_timer_diff > self.time_out_threshold :
                            print("time_out_timer_diff", time_out_timer_diff)
                            self.alarm_flag = False
                            self.time_flag = False
                            self.time_out_flag = False
        else:
            # print("here")
            if not self.time_out_flag:
                self.time_flag = False
                self.alarm_flag = False
                self.time_out_flag = False



