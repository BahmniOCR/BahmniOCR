import cv2
import numpy as np

import deskew
from corner_detector import CornerDetector
from debug_utils import Debuggable


class ImagePreprocessor(Debuggable):
    def __init__(self):
        Debuggable.__init__(self)

    def preprocess_image(self, image):
        """
        Reduce noise

        :param image: image array in numpy
        :return: the processed image
        """
        image = cv2.medianBlur(image, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Binarisation
        processedimage = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processedimage = cv2.medianBlur(processedimage, 3)
        return processedimage

    def deskew_image(self, im, pim):
        cornerdetector = CornerDetector(pim)
        corners = cornerdetector.detect_corners()
        for i in range(0, len(corners)):
            corner = corners[i]
            nextcorner = corners[(i + 1) % len(corners)]
            cv2.circle(im, (corner[0], corner[1]), 25, (210, 210, 210), -1)
            cv2.circle(im, (corner[0], corner[1]), 27, (130, 210, 210), 4)
            cv2.line(im, tuple(corner), tuple(nextcorner), (130, 210, 210), 5)
        self.debug_image(im, cmap='gray')
        deskewer = deskew.Deskewer(im, corners, 1.414)
        transformedimage = deskewer.deskew()
        return transformedimage

    def dilate_erode(self, img):
        """
        Erode and then Dilate image
        """
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(src=img, kernel=kernel, iterations=2)
        dilated_eroded = cv2.dilate(src=eroded, kernel=kernel, iterations=10)
        return dilated_eroded

