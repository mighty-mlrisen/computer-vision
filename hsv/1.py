import cv2


img1 = cv2.imread("/Users/artemmazurenko/mediaProcessingAlgorithms/hsv/test.png",  )

cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)

#cv2.namedWindow("Display window", cv2.WINDOW_FULLSCREEN)

cv2.imshow("Display window", img1)

cv2.waitKey(0)

cv2.destroyAllWindows()


"""
cap = cv2.VideoCapture("practice.mov", cv2.CAP_ANY)

ret, frame = cap.read()
if not(ret):
    0
cv2.imshow('frame', frame)
if cv2.waitKey(1) & 0xFF == 27:
    0
"""

cap = cv2.VideoCapture("practice.mov", cv2.CAP_ANY)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Видео закончилось")
        break

    cv2.imshow("Оригинальное видео", frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
