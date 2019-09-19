# modularized clean.ipynb

import cv2 as cv

from cv_utils import get_grid_mask, verify_grid, find_corners_from_contour, crop_and_warp


def clean(image):
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

        return new_image, corners

    return None, None
