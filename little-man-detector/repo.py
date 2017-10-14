import os
from PIL import Image


class ImageRepo:
    def __init__(self, path):
        self.path = path

    def get_positive_images(self):
        return self.__get_images('positive')

    def get_negative_images(self):
        return self.__get_images('negative')

    def __get_images(self, dataset):
        training = os.path.join(self.path, dataset)
        file_paths = os.listdir(training)
        files = [Image.open(os.path.join(training, f)).resize((20, 30)) for f in file_paths]

        return files
