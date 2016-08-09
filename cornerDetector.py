import numpy as np
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from itertools import compress
from matplotlib import pyplot as plt

class cornerDetector():
	"""Detects page corners in a thresholded image"""

	def __init__(self, image):
		self.__image = image
		self.__width = np.shape(image)[1]
		self.__height = np.shape(image)[0]
		self.detectCorners()

	def detectCorners(self):
		if hasattr(self,'__corners'):
			return self.__corners
		x,y = 0,0
		seeds = [(self.__width/2,0,1,0,0,1), (self.__width,self.__height/2,0,1,-1,0),\
				(self.__width/2,self.__height,-1,0,0,-1), (0,self.__height/2,0,-1,1,0)]
		self.edges = []
		for seed in seeds:
			currEdge = self.findEdge(seed)
			if currEdge==None:
				return [[self.__width,0],[self.__width,self.__height],\
						[0,self.__height],[0,0]]
			self.edges.append(currEdge)
		corners = self.getCorners(self.edges)
		self.__corners = corners
		return corners

	def findEdge(self,seed):
		Xi,Yi,Yli = self.collectEdgePoints(seed)
		plt.scatter(Xi,Yi)
		Xi,Yi = self.getBiggestCluster(Xi,Yi,Yli)
		model = linear_model.RANSACRegressor(linear_model.LinearRegression())
		try:
			model.fit(Xi,Yi)
		except ValueError:
			print("No edge for seed")
			print(seed)
			return None
		return (model.estimator_.coef_[0], model.estimator_.intercept_)

	def collectEdgePoints(self,seed):
		x0,y0,xdir,ydir,xpdir,ypdir = seed
		Xi, Yi, Yli = [],[],[]
		if xdir!=0:
			for x in range(x0-(xdir*self.__width/4),x0+(xdir*self.__width/4),xdir):
				xi,yi = self.dropScanLine(x,y0,xpdir,ypdir)
				Xi.append([xi])
				Yi.append(yi)
				Yli.append([yi])
		if ydir!=0:
			for y in range(y0-(ydir*self.__height/4),y0+(ydir*self.__height/4),ydir):
				xi,yi = self.dropScanLine(x0,y,xpdir,ypdir)
				Xi.append([xi])
				Yi.append(yi)
				Yli.append([xi])
		return Xi, Yi, Yli

	def getBiggestCluster(self,Xi,Yi,Yli):
		clusters = DBSCAN(15).fit_predict(Yli)
		hist,bins = np.histogram(clusters)
		maxi = np.argmax(hist)
		Xi = list(compress(Xi,clusters==self.getIntBetween(bins[maxi],bins[maxi+1])))
		Yi = list(compress(Yi,clusters==self.getIntBetween(bins[maxi],bins[maxi+1])))
		return Xi,Yi

	def getIntBetween(self,num1,num2):
		if(np.isclose((num1*10-10)/10,0)):
			return int(num1)
		else:
			return int(np.floor(num2))

	def dropScanLine(self,x0,y0,xdir,ydir):
		x=x0+xdir
		y=y0+ydir
		while x>=0 and x<self.__width and y>=0 and y<self.__height:
			if self.__image[y,x]==0:
				return (x,y)
			x=x+xdir
			y=y+ydir
		return (x-xdir,y-ydir)

	def getCorners(self,edges):
		corners = []
		n = len(edges)
		for i in range(0,n):
			line1 = edges[i]
			line2 = edges[(i+1)%n]
			corners.append(self.getIntersection(line1,line2))
		return corners

	def getIntersection(self,line1,line2):
		a,c = line1
		b,d = line2
		return [int((d-c)/(a-b)),int((a*d-b*c)/(a-b))]