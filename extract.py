import logging
from pprint import pprint

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from predict import Predictor
from clean import clean
from split import get_cells


def extract(image, predictor):
    try:
        sudoku, _ = clean(image)
        if sudoku is None:
            return None

        cells = get_cells(sudoku)
        digits = predictor.predict(np.reshape(cells, (81, 28, 28, 1)))

        return digits.reshape(9, 9)
    except Exception as e:
        logging.error(e)


if __name__ == '__main__':
    path = 'data/train/image1000.jpg'
    image = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

    predictor = Predictor('models/digit_rec.h5')
    digits = extract(image, predictor)

    pprint(digits)
