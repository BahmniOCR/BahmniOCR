import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sg

from debug_utils import Debuggable
from image_preprocessing import ImagePreprocessor


class Segmenter(Debuggable):
    """Segments text based on white spaces"""

    def __init__(self, image, pim):
        Debuggable.__init__(self)
        # self._debug = True
        self._image = image
        self._width = np.shape(image)[1]
        self._height = np.shape(image)[0]
        self._pim = 255 - pim
        self._threshold = 15
        self._peak_threshold = None
        self._axis = None
        self._slack = 5
        self._segments = []

    def create_segments(self):
        hist = np.sum(self._pim, axis=self._axis)
        smhist = sg.medfilt(hist, 21)
        diffhist = np.diff(smhist)
        self.debug_plot(diffhist)

        peaks = self.get_negative_peaks(diffhist)
        self.log("Negative peaks")
        self.log(peaks)

        peaks = self.merge_nearby_peaks(peaks, diffhist)
        self.log("Merged negative peaks")
        self.log(peaks)

        if len(peaks) <= 1:
            self._segments = []
        else:
            self._segments.insert(0, (0, peaks[0] + self._slack))
            for i in range(1, len(peaks)):
                self.segments.append((peaks[i - 1], peaks[i] + self._slack))
            self._segments.append((peaks[-1], self._height))

    @property
    def segments(self):
        if not self._segments:
            self.create_segments()
        return self._segments

    def get_positive_peaks(self, diffhist):
        return [i for i in range(1, len(diffhist) - 2)
                if diffhist[i - 1] < diffhist[i]
                and diffhist[i + 1] < diffhist[i]
                and diffhist[i] > self._peak_threshold]

    def get_negative_peaks(self, diffhist):
        return [i for i in range(1, len(diffhist) - 1)
                if diffhist[i - 1] > diffhist[i]
                and diffhist[i + 1] > diffhist[i] < -self._peak_threshold]

    def merge_nearby_peaks(self, peaks, diffhist):
        if len(peaks) == 0:
            return peaks
        mergedpeaks = [peaks[i-1] for i in range(1, len(peaks))
                       if diffhist[peaks[i]] - diffhist[peaks[i - 1]] > -self._threshold]
        mergedpeaks.insert(0, peaks[0])
        mergedpeaks.append(peaks[-1])
        return mergedpeaks


class LineSegmenter(Segmenter):

    def __init__(self, image, pim):
        Segmenter.__init__(self, image, pim)
        self._axis = 1
        self._peak_threshold = 255 * 10

    def get_segment_images(self):
        images = []
        for segment in self.segments:
            top, bottom = segment
            images.append(self._image[top:bottom, 0:self._width])
        return images

    def display_segments(self):
        disp_image = copy.copy(self._image)
        for segment in self.segments:
            top, bottom = segment
            cv2.line(disp_image, (0, top), (self._width, top), (0, 0, 255), 1)
            cv2.line(disp_image, (0, bottom), (self._width, bottom), (0, 0, 255), 1)
        self.debug_image(disp_image)


class WordSegmenter(Segmenter):

    def __init__(self, image, pim):
        Segmenter.__init__(self, image, pim)
        self._debug = True
        self._axis = 0
        self._peak_threshold = 255 * 4
        self.ip = ImagePreprocessor()
        self._pim = self.ip.dilate_erode(255 - pim)
        self.debug_image(self._pim, cmap='gray')

    def display_segments(self):
        for segment in self.segments:
            left, right = segment
            cv2.line(self._image, (left, 0), (left, self._height), (0, 0, 255), 1)
            cv2.line(self._image, (right, 0), (right, self._height), (0, 0, 255), 1)
        self.debug_image(self._image)
