from skimage.feature import hog
from skimage import data, color, exposure
import matplotlib.pyplot as plt
from numpy import asarray
from sklearn import svm
from PIL import Image
import os

class LittleManDetectorTrainer:

    def __init__(self, repo):
        self.repo = repo

    def train(self):
        positive_samples = self.repo.get_positive_images()
        negative_samples = self.repo.get_negative_images()

        X = []
        Y = []

        for negative in negative_samples:
            collapsed = asarray(negative)
            image = color.rgb2gray(collapsed)

            fd, hog_image = self.hog(image)

            #self.plot_hog(image, hog_image)
            X.append(fd)
            Y.append(0)

        for positive in positive_samples:
            collapsed = asarray(positive)
            image = color.rgb2gray(collapsed)

            fd, hog_image = self.hog(image)

            #self.plot_hog(image, hog_image)

            X.append(fd)
            Y.append(1)

        model = svm.SVC(kernel='linear', probability=True)
        model.fit(X, Y)
        score = model.score(X, Y)

        return model

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

    def hog(self, image):
        return hog(image, orientations=9, pixels_per_cell=(6, 6), block_norm='L2',
            cells_per_block=(3, 3), visualise=True)