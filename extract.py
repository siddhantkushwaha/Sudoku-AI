import logging

import numpy as np

from clean import clean
from split import get_cells


def extract(image, predictor):
    try:
        sudoku = clean(image)
        if sudoku is None:
            return None

        cells = get_cells(sudoku)
        digits = predictor.predict(np.reshape(cells, (81, 28, 28, 1)))

        return digits.reshape(9, 9)
    except Exception as e:
        logging.error(e)
