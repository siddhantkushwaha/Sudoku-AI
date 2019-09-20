import cv2 as cv

from clean_v2 import clean
from cv_utils import get_grid_mask


def main():
    cap = cv.VideoCapture(0)

    ref_sudoku = cv.imread('data/ref_sudoku.jpg')
    ref_mask, _, _ = get_grid_mask(ref_sudoku)

    while True:

        ret, frame = cap.read()

        _, corners = clean(frame, ref_mask)
        if corners is not None:
            cv.rectangle(img=frame, pt1=(corners[0][0], corners[0][1]),
                         pt2=(corners[2][0], corners[2][1]), color=(0, 255, 0),
                         thickness=1)

            cv.putText(img=frame, text='TL', color=(0, 0, 255), org=(corners[0][0], corners[0][1]),
                       fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)

            cv.putText(img=frame, text='BR', color=(0, 0, 255), org=(corners[2][0], corners[2][1]),
                       fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
