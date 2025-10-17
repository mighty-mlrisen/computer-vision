import cv2 as cv

def task9():
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
    cv.namedWindow("PhoneWebcam", cv.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow("PhoneCam", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
cv.destroyAllWindows()

task9()