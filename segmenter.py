import numpy as np

class segmenter():
	"""Segments page as a precursor to OCR process"""

	def __init__(self,image,pim,width):
		self.__image = image
		self.__pim = pim
		self.__width = width
		self.__threshold = 255*width*0.7
		self.segmentPage()

	def segmentPage(self):
		hist = np.sum(self.__pim,1)
		peaks = self.getPeaks(hist)
		segments = self.getSegments(peaks,hist)

	def getPeaks(self,hist):
		incr,l,hills,hill = False,len(hist),[],[]
		for i in range(l):
			if (not incr) and hist[i]>=threshold:
				incr = True
			if incr and hist[i]>=threshold:
				hill.append(i)
			elif incr and hist[i]<threshold:
				incr = False
				hills.append(hill)
				hill = []
		peaks = self.getPeaksFromHills(hist,hills)
		return peaks

	def getPeaksFromHills(self,hist,hills):
		