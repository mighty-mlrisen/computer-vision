import cv2 as cv


def task8():
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
    cv.namedWindow("Webcam", cv.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        #Извлекаем значения BGR каналов центрального пикселя:
        b, g, r = frame[cy, cx]
        #print(frame[cy, cx])
        if r >= g and r >= b:
            color = (0, 0, 255)
        elif g >= r and g >= b:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        rectG_w = 120
        rectG_h = 20
        rectV_w = 20
        rectV_h = 120

        cv.rectangle(frame, (cx - rectG_w // 2, cy - rectG_h // 2),
                            (cx + rectG_w // 2, cy + rectG_h // 2),
                            color, -1)
        cv.rectangle(frame, (cx - rectV_w // 2, cy - rectV_h // 2),
                            (cx + rectV_w // 2, cy + rectV_h // 2),
                            color, -1)

        cv.imshow("Webcam", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv.destroyAllWindows()
task8()