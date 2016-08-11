import numpy as np
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from itertools import compress
from matplotlib import pyplot as plt


# noinspection SpellCheckingInspection
class CornerDetector:
    """Detects page corners in a thresholded image"""

    def __init__(self, image):
        self.__image = image
        self.__width = np.shape(image)[1]
        self.__height = np.shape(image)[0]
        self.detect_corners()

    def detect_corners(self):
        if hasattr(self, '__corners'):
            return self.__corners
        x, y = 0, 0
        seeds = [(self.__width / 2, 0, 1, 0, 0, 1), (self.__width, self.__height / 2, 0, 1, -1, 0), \
                 (self.__width / 2, self.__height, -1, 0, 0, -1), (0, self.__height / 2, 0, -1, 1, 0)]
        self.edges = []
        for seed in seeds:
            current_edge = self.find_edge(seed)
            if current_edge is None:
                return [[self.__width, 0], [self.__width, self.__height], \
                        [0, self.__height], [0, 0]]
            self.edges.append(current_edge)
        corners = self.get_corners()
        self.__corners = corners
        return corners

    def find_edge(self, seed):
        xi, yi, yli = self.collect_edge_points(seed)
        plt.scatter(xi, yi)
        xi, yi = self.get_biggest_cluster(xi, yi, yli)
        model = linear_model.RANSACRegressor(linear_model.LinearRegression())
        try:
            model.fit(xi, yi)
        except ValueError:
            print("No edge for seed")
            print(seed)
            return None
        return model.estimator_.coef_[0], model.estimator_.intercept_

    def collect_edge_points(self, seed):
        x0, y0, xdir, ydir, xpdir, ypdir = seed
        xi, yi, yli = [], [], []
        if xdir != 0:
            for x in range(x0 - (xdir * self.__width / 4), x0 + (xdir * self.__width / 4), xdir):
                xj, yj = self.drop_scan_line(x, y0, xpdir, ypdir)
                xi.append([xj])
                yi.append(yj)
                yli.append([yj])
        if ydir != 0:
            for y in range(y0 - (ydir * self.__height / 4), y0 + (ydir * self.__height / 4), ydir):
                xj, yj = self.drop_scan_line(x0, y, xpdir, ypdir)
                xi.append([xj])
                yi.append(yj)
                yli.append([xj])
        return xi, yi, yli

    @staticmethod
    def get_biggest_cluster(xi, yi, yli):
        clusters = DBSCAN(15).fit_predict(yli)
        hist, bins = np.histogram(clusters)
        maxi = np.argmax(hist)
        xi = list(compress(xi, clusters == CornerDetector.get_int_between(bins[maxi], bins[maxi + 1])))
        yi = list(compress(yi, clusters == CornerDetector.get_int_between(bins[maxi], bins[maxi + 1])))
        return xi, yi

    @staticmethod
    def get_int_between(num1, num2):
        if np.isclose((num1 * 10 - 10) / 10, 0):
            return int(num1)
        else:
            return int(np.floor(num2))

    def drop_scan_line(self, x0, y0, xdir, ydir):
        x = x0 + xdir
        y = y0 + ydir
        while 0 <= x < self.__width and 0 <= y < self.__height:
            if self.__image[y, x] == 0:
                return x, y
            x = x + xdir
            y = y + ydir
        return x - xdir, y - ydir

    def get_corners(self):
        corners = []
        n = len(self.edges)
        for i in range(0, n):
            line1 = self.edges[i]
            line2 = self.edges[(i + 1) % n]
            corners.append(CornerDetector.get_intersection(line1, line2))
        return corners

    @staticmethod
    def get_intersection(line1, line2):
        a, c = line1
        b, d = line2
        return [int((d - c) / (a - b)), int((a * d - b * c) / (a - b))]
