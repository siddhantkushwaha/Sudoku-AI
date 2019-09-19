from pprint import pprint

import cv2 as cv
import matplotlib.pyplot as plt

from extract import extract
from predict import Predictor

if __name__ == '__main__':
    path = 'data/train/image1000.jpg'
    image = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

    predictor = Predictor('models/digit_rec.h5')
    digits = extract(image, predictor)

    pprint(digits)
