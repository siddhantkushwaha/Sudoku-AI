import numpy as np

from clean import clean
from split import get_cells


def extract(image, predictor):
    sudoku = clean(image)
    cells = get_cells(sudoku)
    digits = predictor.predict(np.reshape(cells, (81, 28, 28, 1)))

    return digits.reshape(9, 9)
