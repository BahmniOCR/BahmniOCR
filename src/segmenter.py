import numpy as np
from scipy import signal as sg
import cv2
from matplotlib import pyplot as plt
import copy
from image_preprocessing import debugPlot
from image_preprocessing import debugImage


class Segmenter:
    """Segments text based on white spaces"""

    def __init__(self, image, pim):
        self._image = image
        self._width = np.shape(image)[1]
        self._height = np.shape(image)[0]
        self._pim = 255 - pim
        debugImage(self._pim)
        self._threshold = 15
        self._slack = 5
        self._segments = []

    def create_segments(self):
        hist = np.sum(self._pim, axis=self._axis)
        smhist = sg.medfilt(hist, 21)
        diffhist = np.diff(smhist)
        debugPlot(diffhist)
        peaks = self.get_negative_peaks(diffhist)
        print("Length of negative peaks")
        print(peaks)
        print(len(peaks))
        peaks = self.merge_nearby_peaks(peaks, diffhist)
        print("Length of merged negative peaks")
        print(peaks)
        print(len(peaks))

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


class WordSegmenter(Segmenter):

    def __init__(self, image, pim):
        Segmenter.__init__(self, image, pim)
        self._axis = 0
        self._peak_threshold = 255 * 4

    def display_segments(self):
        for segment in self.segments:
            left, right = segment
            cv2.line(self._image, (left, 0), (left, self._height), (0, 0, 255), 1)
            cv2.line(self._image, (right, 0), (right, self._height), (0, 0, 255), 1)
        plt.imshow(self._image)
        plt.show()


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
        plt.imshow(disp_image)
        plt.show()
