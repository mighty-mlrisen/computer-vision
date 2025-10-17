
import cv2 as cv

def task5():
    img = cv.imread("media/basketball.png")
    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    cv.namedWindow("basketball BGR", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("basketball HSV", cv.WINDOW_AUTOSIZE)

    cv.imshow("basketball BGR", img)
    cv.imshow("basketball HSV", img_HSV)

    cv.waitKey(0)
    cv.destroyAllWindows()

task5()