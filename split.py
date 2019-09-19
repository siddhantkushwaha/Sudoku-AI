import numpy as np
import cv2 as cv


def get_cell(img, i, j):
    width = img.shape[0] // 9
    return img[i * width:(i + 1) * width, j * width:(j + 1) * width]


def get_cells(img):
    cells = []
    for i in range(9):
        cells.append([])
        for j in range(9):
            cell = get_cell(img, i, j)

            cell = cv.resize(cell, (28, 28))
            cell = cv.cvtColor(cell, cv.COLOR_RGB2GRAY)
            cell = cell / 255.0

            cells[-1].append(cell)

    return np.asarray(cells)
