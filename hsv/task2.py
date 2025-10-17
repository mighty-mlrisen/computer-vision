import cv2 as cv

def task2():
    image1 = cv.imread("media/ball.jpg", cv.IMREAD_GRAYSCALE)
    image2 = cv.imread("media/ball.png", cv.IMREAD_COLOR)
    image3 = cv.imread("media/ball.jpeg", cv.IMREAD_REDUCED_COLOR_8)

    cv.namedWindow("ball.jpg", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("ball.png", cv.WINDOW_KEEPRATIO)
    cv.namedWindow("ball.jpeg", cv.WINDOW_NORMAL)

    cv.imshow("ball.jpg", image1)
    cv.imshow("ball.png", image2)         
    cv.imshow("ball.jpeg", image3)
 
    cv.waitKey(0)
    cv.destroyAllWindows()

task2()