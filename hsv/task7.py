import cv2 as cv


def task7():
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
    cv.namedWindow("Webcam", cv.WINDOW_AUTOSIZE)

    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter("media/webcam_output.mp4", fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv.imshow("Webcam", frame)
        out.write(frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

task7()