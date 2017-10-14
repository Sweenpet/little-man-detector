from little_man_detector_trainer import LittleManDetectorTrainer
from little_man_detector import LittleManDetector
from repo import ImageRepo
import os

def detect_little_men():
    print 'detecting little men'

def train():
    path = os.path.join(get_parent_directory(os.getcwd()), 'images') + '\\'
    repo = ImageRepo(path)
    trainer= LittleManDetectorTrainer(repo)
    return trainer.train()

def detect():
    image_path = "C:\\projects\python\\little-man-detector\\images\\test\\test_0.jpg"
    model = train()
    detector = LittleManDetector(model)
    detector.detect(image_path)

def get_parent_directory(cd):
    return os.path.abspath(os.path.join(cd, os.pardir))


if __name__ == '__main__':
    detect()
