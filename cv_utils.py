import operator

import numpy as np
import cv2 as cv


def isolate_lines(src, structuring_element):
    cv.erode(src, structuring_element, src, (-1, -1))  # makes white spots smaller
    cv.dilate(src, structuring_element, src, (-1, -1))


def get_grid_mask(image):
    # convert to grayscale
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    max_threshold_value = 255
    block_size = 15
    threshold_constant = 0

    # Filter image
    filtered = cv.adaptiveThreshold(~grayscale, max_threshold_value, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                    block_size, threshold_constant)

    # HORIZONTAL AND VERTICAL LINE ISOLATION
    # To isolate the vertical and horizontal lines,
    #
    # 1. Set a scale.
    # 2. Create a structuring element.
    # 3. Isolate the lines by eroding and then dilating the image.

    scale = 15

    # Isolate horizontal and vertical lines using morphological operations
    horizontal = filtered.copy()
    vertical = filtered.copy()

    horizontal_size = int(horizontal.shape[1] / scale)
    horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    isolate_lines(horizontal, horizontal_structure)

    vertical_size = int(vertical.shape[0] / scale)
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    isolate_lines(vertical, vertical_structure)

    # Create an image mask with just the horizontal
    # and vertical lines in the image. Then find
    # all contours in the mask.

    mask = horizontal + vertical

    return mask, horizontal, vertical


def verify_grid(contour, intersections):
    # min table area to be considered a table
    min_table_area = 50
    # epsilon value for contour approximation
    epsilon = 3

    area = cv.contourArea(contour)

    if area < min_table_area:
        return None

    # approxPolyDP approximates a polygonal curve within the specified precision
    curve = cv.approxPolyDP(contour, epsilon, True)

    # boundingRect calculates the bounding rectangle of a point set (eg. a curve)
    rect = cv.boundingRect(curve)  # format of each rect: x, y, w, h

    # Finds the number of joints in each region of interest (ROI)
    # Format is in row-column order (as finding the ROI involves numpy arrays)
    # format: image_mat[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w]
    possible_table_region = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    possible_table_joints, _ = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Determines the number of table joints in the image
    # If less than 5 table joints, then the image
    # is likely not a table
    if len(possible_table_joints) < 5:
        return None

    return rect


def find_corners_from_contour(polygon):
    # Finds the 4 extreme corners of the contour given.

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
    # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value

    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1]
                                 for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1]
                                    for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1]
                                  for pt in polygon]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def distance_between(p1, p2):
    # Returns the scalar distance between two points
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
    # Crops and warps a rectangular section from an image into a square of similar size.

    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[
                                                         0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right,
                    bottom_left], dtype='float32')

    b = distance_between(top_left, bottom_left)
    l = distance_between(bottom_left, bottom_right)

    side = max(l, b)
    # Describe a square with side of the calculated length and breadth, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                    [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    return cv.warpPerspective(img, m, (int(side), int(side)))
