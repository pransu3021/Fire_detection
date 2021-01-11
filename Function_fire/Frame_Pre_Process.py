"""
It will preproess the image and adjust the blur and gamma of the image.
"""

import cv2
import numpy as np
class PreProcess:
    """
    It will return the auto corrected gama value frame
    """

    def __init__(self,blur_thresh,gamma_thresh):
        self.image = None
        self.blurThreshold = blur_thresh
        self.Gamma_threshold = gamma_thresh
        self.exp_Gamma_intensity = 120
        # self.Gamma = AutoGammaControl()

    def blur(self):
        return cv2.Laplacian(self.image, cv2.CV_64F).var()

    def gamma_correction(self,img):
        # Extract intensity component of the image
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = YCrCb[:, :, 0]
        M, N = img.shape[:2]
        mean_in = np.sum(Y / (M * N))
        t = (mean_in - self.exp_Gamma_intensity) / self.exp_Gamma_intensity
        # Process image for gamma correction

        if t < self.Gamma_threshold:  # Dimmed Image
            # print(name + ": Dimmed")
            result = self.process_dimmed(Y)
            YCrCb[:, :, 0] = result
            y_new = YCrCb
            M, N = img.shape[:2]
            mean_in = np.sum(y_new / (M * N))
            thresh = (mean_in - self.exp_Gamma_intensity) / self.exp_Gamma_intensity
            img_output = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)

        elif t > self.Gamma_threshold:  # Bright Image
            # print(name + ": Bright Image")
            result = self.process_bright(Y)
            YCrCb[:, :, 0] = result
            y_new = YCrCb
            M, N = img.shape[:2]
            mean_in = np.sum(y_new / (M * N))
            thresh = (mean_in - self.exp_Gamma_intensity) / self.exp_Gamma_intensity
            img_output = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)

        else:
            img_output = img
            thresh = t
        return img_output

    def process_bright(self, img):
        img_negative = 255 - img
        agcwd = self.image_agcwd(img_negative, a=0.25, truncated_cdf=False)
        reversed = 255 - agcwd
        return reversed

    def process_dimmed(self, img):
        agcwd = self.image_agcwd(img, a=0.75, truncated_cdf=True)
        return agcwd

    # @staticmethod
    def image_agcwd(self, img, a=0.25, truncated_cdf=False):
        h, w = img.shape[:2]
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        prob_normalized = hist / hist.sum()

        unique_intensity = np.unique(img)
        # print(unique_intensity)
        intensity_max = unique_intensity.max()
        intensity_min = unique_intensity.min()
        prob_min = prob_normalized.min()
        prob_max = prob_normalized.max()
        # print(unique_intensity.max(),unique_intensity.min(),prob_normalized.min(),prob_normalized.max())

        pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
        pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0] ** a)
        pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0]) ** a))
        prob_normalized_wd = pn_temp / pn_temp.sum()  # normalize to [0,1]
        cdf_prob_normalized_wd = prob_normalized_wd.cumsum()

        if truncated_cdf:
            inverse_cdf = np.maximum(0.6, 1 - cdf_prob_normalized_wd)
            # print("inv: ",inverse_cdf)
        else:
            inverse_cdf = 1 - cdf_prob_normalized_wd
            # print("inv: ", inverse_cdf)

        img_new = img.copy()
        for i in unique_intensity:
            img_new[img == i] = np.round(255 * (i / 255) ** inverse_cdf[i])
        # print(img_new)
        # cv2.imwrite('try1111.jpg', img_new)
        return img_new

def equalise_img(frame, blur_thresh,gamma_thresh):
    # blur_thresh = 100
    # gamma_thresh = 100
    if blur_thresh > 0 or gamma_thresh> 0:
        preprocess = PreProcess(blur_thresh, gamma_thresh)
        frame = preprocess.gamma_correction(frame)
        return frame
    else:
        return frame

