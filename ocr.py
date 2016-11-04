import cv2
from matplotlib import pyplot as plt

import CornerDetector as cd
import Deskew
import Segmenter as seg


def preprocess_image(image):
    image = cv2.medianBlur(image, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    processedimage = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processedimage = cv2.medianBlur(processedimage, 3)
    return processedimage


def deskew_image(im, pim):
    cornerdetector = cd.CornerDetector(pim)
    corners = cornerdetector.detect_corners()
    for i in range(0, len(corners)):
        corner = corners[i]
        nextcorner = corners[(i + 1) % len(corners)]
        cv2.circle(im, (corner[0], corner[1]), 25, (210, 210, 210), -1)
        cv2.circle(im, (corner[0], corner[1]), 27, (130, 210, 210), 4)
        cv2.line(im, tuple(corner), tuple(nextcorner), (130, 210, 210), 5)
    plt.imshow(im, cmap='gray')
    plt.show()
    deskewer = Deskew.Deskewer(im, corners, 1.414)
    transformedimage = deskewer.deskew()
    return transformedimage


img = cv2.cvtColor(cv2.imread('form5.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
im = preprocess_image(img)
img = deskew_image(img, im)
plt.imshow(img)
plt.show()
pim = preprocess_image(img)
segmenter = seg.Segmenter(img, pim)
segmenter.display_segments()
