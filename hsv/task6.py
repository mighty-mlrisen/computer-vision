import cv2 as cv

def task6():
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
    cv.namedWindow("Webcam", cv.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        rectG_w = 170
        rectG_h = 30
        rectV_w = 30
        rectV_h = 170

        cv.rectangle(frame, (cx - rectG_w // 2, cy - rectG_h // 2),
                            (cx + rectG_w // 2, cy + rectG_h // 2),
                            (0, 0, 255), 2)
        cv.rectangle(frame, (cx - rectV_w // 2, cy - rectV_h // 2),
                            (cx + rectV_w // 2, cy + rectV_h // 2),
                            (0, 0, 255), 2)

        cv.imshow("Webcam", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

task6()