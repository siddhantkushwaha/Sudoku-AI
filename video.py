import cv2 as cv

from clean import clean

if __name__ == '__main__':

    k = 1
    cap = cv.VideoCapture(0)

    while (True):

        ret, frame = cap.read()

        cleaned, corners = clean(frame)
        if cleaned is not None:
            print(k, cleaned.shape, corners)

            cv.rectangle(img=frame, pt1=(corners[0][0], corners[0][1]),
                         pt2=(corners[2][0], corners[2][1]), color=(0, 255, 0),
                         thickness=1)

            cv.putText(img=frame, text='TL', color=(0, 0, 255), org=(corners[0][0], corners[0][1]),
                       fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)

            cv.putText(img=frame, text='BR', color=(0, 0, 255), org=(corners[2][0], corners[2][1]),
                       fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)

            k += 1

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
