# modularized clean_v2.ipynb
import os

import numpy as np
import cv2 as cv

from cv_utils import get_grid_mask, find_corners_from_contour, crop_and_warp, verify_grid


def find_similarity(mask, ref_idx):
    score = 0
    for i, j in ref_idx:
        region = mask[i - 8:i + 8, j - 8: j + 8]
        if np.sum(region) > 0:
            score += 1
    return score


def clean(image, ref_idx, cutoff=0.85):
    mask, h, v = get_grid_mask(image)

    # Find intersections between the lines to determine if the intersections are grid joints.
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # if a exists, it has to be the biggest polygon
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    intersections = cv.bitwise_and(h, v)

    for grid_number, contour in enumerate(contours):

        # verify that Region of Interest (ROI) is a table
        rect = verify_grid(contour, intersections)
        if rect is None:
            continue

        corners = find_corners_from_contour(contour)
        new_image = crop_and_warp(image, corners)
        new_image = cv.resize(new_image, (512, 512))

        new_mask, _, _ = get_grid_mask(new_image)

        n = find_similarity(new_mask, ref_idx)
        if n / ref_idx.shape[0] > cutoff:
            return new_image, corners

    return None, None


def main():
    ref_sudoku = cv.imread('data/ref_sudoku.jpg')
    ref_mask, _, _ = get_grid_mask(ref_sudoku)
    ref_idx = np.argwhere(ref_mask == 255.0)

    for path in os.listdir('data/train'):
        if '.jpg' not in path:
            continue

        fpath = f'data/train/{path}'
        print(fpath)

        image = cv.imread(fpath)
        sudoku, _ = clean(image, ref_idx)

        if sudoku is not None:
            cv.imwrite(f'out/{path}', sudoku)


if __name__ == '__main__':
    main()
