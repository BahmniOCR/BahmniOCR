import math

import cv2
import numpy as np


# noinspection SpellCheckingInspection
class Deskewer:
    """Deskews page image given image, skewed corner point coordinates and aspect ratio"""
    __aspect = 1.414

    def __init__(self, image, corners, aspect):
        self.__image = image
        self.__corners = corners
        self.__aspect = aspect
        self.deskew()

    def deskew(self):
        if hasattr(self, '__transImage'):
            return self.__transImage
        transformedcorners = self.get_transformed_coords()
        floatcorners = np.float32(self.__corners)
        floattransformedcorners = np.float32(transformedcorners)
        m = cv2.getPerspectiveTransform(floatcorners, floattransformedcorners)
        transformedimage = cv2.warpPerspective(self.__image, m, (self.__w, self.__h))
        self.__transformedimage = transformedimage
        return transformedimage

    def get_transformed_coords(self):
        self.get_area()
        w = int(math.sqrt(self.__area / self.__aspect))
        h = int(math.sqrt(self.__area * self.__aspect))
        self.__w = w
        self.__h = h
        return [[w, 0], [w, h], [0, h], [0, 0]]

    def get_area(self):
        if hasattr(self, '__area'):
            return self.__area
        area = 0
        n = len(self.__corners)
        for i in range(0, n):
            corner1 = self.__corners[i]
            corner2 = self.__corners[(i + 1) % n]
            area += corner1[0] * corner2[1] - corner2[0] * corner1[1]
        self.__area = abs(area / 2)
        return self.__area
