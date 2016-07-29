import numpy as np
import math
import cv2

class deskewer():
	"""Deskews page image given image, skewed corner point coordinates and aspect ratio"""
	__image = None
	__corners = []
	__aspect = 1.414
	__area = -1
	__w = -1
	__h = -1
	__transImage = None

	def __init__(self, image, corners, aspect):
		self.__image = image
		self.__corners = corners
		self.__aspect = aspect
		self.deskew()

	def deskew(self):
		if self.__transImage!=None:
			return self.__transImage
		transCorners = self.getTransformedCoords()
		fCorners = np.float32(self.__corners)
		fTransCorners = np.float32(transCorners)
		M = cv2.getPerspectiveTransform(fCorners,fTransCorners)
		transImage = cv2.warpPerspective(self.__image,M,(self.__w,self.__h))
		self.__transImage = transImage
		return transImage

	def getTransformedCoords(self):
		self.getArea()
		w = int(math.sqrt(self.__area/self.__aspect))
		h = int(math.sqrt(self.__area*self.__aspect))
		self.__w = w
		self.__h = h
		return [[w,0],[w,h],[0,h],[0,0]]

	def getArea(self):
		if self.__area>0:
			return self.__area
		area = 0
		n = len(self.__corners)
		for i in range(0,n):
			corner1 = self.__corners[i]
			corner2 = self.__corners[(i+1)%n]
			area += corner1[0]*corner2[1]-corner2[0]*corner1[1]
		self.__area = abs(area/2)
		return self.__area