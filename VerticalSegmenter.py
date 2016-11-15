import numpy as np
from scipy import signal as sg
import cv2
from matplotlib import pyplot as plt


# noinspection SpellCheckingInspection
class VerticalSegmenter:
    """Segments page as a precursor to OCR process"""

    def __init__(self, image, pim):
        self.__image = image
        self.__width = np.shape(image)[1]
        self.__height = np.shape(image)[0]
        self.__pim = pim
        self.__threshold = 15
        self.__slack = 5
        self.get_segments()

    def display_segments(self):
        if not hasattr(self, '__segments'):
            self.get_segments()
        for segment in self.__segments:
            left, right = segment
            cv2.line(self.__image, (left, 0), (left, self.__height), (0, 0, 255), 1)
            cv2.line(self.__image, (right, 0), (right, self.__height), (0, 0, 255), 1)
        plt.imshow(self.__image)
        plt.show()

    def get_segments(self):
        if hasattr(self, '__segments'):
            return self.__segments
        hist = np.sum(self.__pim, 0)
        smhist = sg.medfilt(hist, 21)
        diffhist = np.diff(smhist)
        peaks = self.get_peaks(diffhist)
        peaks = self.merge_nearby_peaks(peaks)
        if len(peaks) <= 1:
            return []
        self.__segments = [(peaks[i - 1], peaks[i] + self.__slack) for i in range(1, len(peaks))]
        self.__segments.insert(0, (0, peaks[0] + self.__slack))
        self.__segments.append((peaks[-1], self.__height))
        return self.__segments

    def get_peaks(self, diffhist):
        peaks = [i for i in range(1, len(diffhist) - 2)
                 if diffhist[i - 1] < diffhist[i]
                 and diffhist[i + 1] < diffhist[i]
                 and diffhist[i] > 255 * 10]
        mergedpeaks = self.merge_nearby_peaks(peaks)
        return mergedpeaks

    def merge_nearby_peaks(self, peaks):
        if len(peaks) == 0:
            return peaks
        mergedpeaks = [peaks[i - 1] for i in range(1, len(peaks))
                       if peaks[i] - peaks[i - 1] > self.__threshold]
        mergedpeaks.append(peaks[-1])
        return mergedpeaks
