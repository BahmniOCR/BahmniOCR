import numpy as np
import cv2
import cornerDetector as cd
import deskew
from matplotlib import pyplot as plt
import segmenter as seg

def preprocessImage(img):
	img = cv2.medianBlur(img,3)
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	im = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	im = cv2.medianBlur(im,3)
	return im

def deskewImage(im,pim):
	corD = cd.cornerDetector(pim)
	corners = corD.detectCorners()
	for i in range(0,len(corners)):
		corner = corners[i]
		ncorner = corners[(i+1)%len(corners)]
		cv2.circle(im,(corner[0],corner[1]),25,(0,255,0),-1)
		cv2.line(im,tuple(corner), tuple(ncorner), (0,0,255),5)
	plt.imshow(im, cmap='gray')
	plt.show()
	deSk = deskew.deskewer(im, corners, 1.414)
	transImg = deSk.deskew()
	return transImg

def getTextRows(im,pim):
	hist = np.sum(pim,1)
	return 1

img = cv2.cvtColor(cv2.imread('form4.jpg'), cv2.COLOR_BGR2RGB)
im = preprocessImage(img)
img = deskewImage(img,im)
pim = preprocessImage(img)
segm = seg.segmenter(img,pim)
segm.displaySegments()