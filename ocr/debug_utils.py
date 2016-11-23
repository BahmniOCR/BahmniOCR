import numpy as np
from matplotlib import pyplot as plt


class Debuggable:
    """
    The following statement must be added in subclasses to enable debugging.

    self._debug = True
    """
    def __init__(self):
        self._debug = False

    def debug_plot(self, array):
        if self._debug:
            plt.plot(np.arange(0, len(array), 1), array)
            plt.show()

    def debug_image(self, image, cmap=None):
        if self._debug:
            plt.imshow(image, cmap=cmap) if cmap else plt.imshow(image)
            plt.show()

    def log(self, msg):
        if self._debug:
            print msg
