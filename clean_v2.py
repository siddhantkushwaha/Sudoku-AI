# modularized clean_v2.ipynb
import os

import numpy as np
import cv2 as cv

from cv_utils import get_grid_mask, find_corners_from_contour, crop_and_warp, verify_grid


def clean(image, ref_mask, cutoff=0.5):
    d = np.argwhere(ref_mask == 255.0).shape[0]

    mask, h, v = get_grid_mask(image)

    # Find intersections between the lines to determine if the intersections are grid joints.
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # if a exists, it has to be the biggest polygon
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    intersections = cv.bitwise_and(h, v)

    count = 0
    for grid_number, contour in enumerate(contours):

        # verify that Region of Interest (ROI) is a table
        rect = verify_grid(contour, intersections)
        if rect is None:
            continue

        corners = find_corners_from_contour(contour)
        new_image = crop_and_warp(image, corners)
        new_image = cv.resize(new_image, (512, 512))

        new_mask, _, _ = get_grid_mask(new_image)

        diff = np.bitwise_and(new_mask, ref_mask)
        n = np.argwhere(diff == 255.0).shape[0]

        if n / d > cutoff:
            count += 1

    return count


if __name__ == '__main__':

    ic = 0
    sc = 0

    ref_sudoku = cv.imread('data/ref_sudoku.jpg')
    ref_mask, _, _ = get_grid_mask(ref_sudoku)

    for path in os.listdir('data/train'):
        if not '.jpg' in path:
            continue

        fpath = f'data/train/{path}'

        ic += 1
        image = cv.imread(fpath)

        res = clean(image, ref_mask, cutoff=0.43)

        if res != 1:
            print(fpath, res)

        sc += res

        # print(fpath, res)

    print(ic, sc)
