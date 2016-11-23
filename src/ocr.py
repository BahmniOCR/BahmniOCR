import cv2
import numpy as np
from matplotlib import pyplot as plt

from segmenter import LineSegmenter, WordSegmenter
from image_preprocessing import preprocess_image, deskew_image

# Display input image
img = cv2.cvtColor(cv2.imread('../resources/form5.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Display preprocessed and deskewed image
im = preprocess_image(img)
img = deskew_image(img, im)
plt.imshow(img)
plt.show()

# Display lines recognized from the page.
pim = preprocess_image(img)
segmenter = LineSegmenter(img, pim)
segmenter.display_segments()

# Display line segments
seg_images = segmenter.get_segment_images()
for seg_image in seg_images:
    pseg_image = preprocess_image(seg_image)
    word_segmenter = WordSegmenter(seg_image, pseg_image)
    word_segmenter.display_segments()