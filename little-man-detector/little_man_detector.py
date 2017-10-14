from PIL import Image
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data, color, exposure
import time as time
from skimage.transform import pyramid_gaussian


class LittleManDetector:

    def __init__(self, model):
        self.model = model

    def detect(self, image_path):
        image = Image.open(image_path)

        collapsed = np.asarray(image)
        image = color.rgb2gray(collapsed)

        rows, cols = image.shape
        pyramid = tuple(pyramid_gaussian(image, downscale=1.5))

        flattened_columns = cols + cols // 2 + 1

        composite_image = np.zeros((rows, flattened_columns ), dtype=np.double)

        composite_image[:rows, :cols] = pyramid[0]

        for p in pyramid[0:]:
            self.process(p)

        plt.show()

    def process(self, pyramid_image):
        (winW, winH) = (20, 30)

        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in self.sliding_window(pyramid_image, stepSize=6, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            im = Image.fromarray(window)
            resized = im.resize((20, 30))

            collapsed = np.asarray(resized)

            fd, hog_image = hog(collapsed, orientations=9, pixels_per_cell=(6, 6),
                                cells_per_block=(3, 3), block_norm='L2', visualise=True)

            # since we do not have a classifier, we'll just draw the window
            clone = pyramid_image.copy()

            square = (0, 0, 255)

            cv2.rectangle(clone, (x, y), (x + winW, y + winH), square, 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)

            prob = self.model.predict_proba([fd])
            print(prob)

            prediction = self.model.predict([fd])
            if prediction[0] == 1:
                self.plot_hog(collapsed, hog_image)

            time.sleep(0.025)

    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in xrange(0, image.shape[0], stepSize):
            for x in xrange(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def plot_hog(self, image, hog_image):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        ax1.axis('off')
        ax1.imshow(image)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap='gray')
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.show()