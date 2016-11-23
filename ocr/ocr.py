import cv2

from image_preprocessing import ImagePreprocessor
from debug_utils import Debuggable
from segmenter import LineSegmenter, WordSegmenter


class OCR(Debuggable):
    def __init__(self):
        Debuggable.__init__(self)

    def main(self):
        # Display input image
        img = cv2.cvtColor(cv2.imread('../resources/form5.jpg'), cv2.COLOR_BGR2RGB)
        self.debug_image(img)

        ip = ImagePreprocessor()
        # Display preprocessed and deskewed image
        im = ip.preprocess_image(img)
        img = ip.deskew_image(img, im)
        self.debug_image(img)

        # Display lines recognized from the page.
        pim = ip.preprocess_image(img)
        segmenter = LineSegmenter(img, pim)
        segmenter.display_segments()

        # Display line segments
        seg_images = segmenter.get_segment_images()
        for seg_image in seg_images:
            pseg_image = ip.preprocess_image(seg_image)
            word_segmenter = WordSegmenter(seg_image, pseg_image)
            word_segmenter.display_segments()


OCR().main()
