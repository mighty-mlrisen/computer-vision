import cv2 as cv


def show_video(path, name, size="", color=""):
    cap = cv.VideoCapture(path)
    cv.namedWindow(name,cv.WINDOW_NORMAL)
    target_size = None
    match size:
        case "small":
            target_size = (640, 360)
        case "medium":
            target_size = (960, 540)
        case "big":
            target_size = (1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.resize(frame, target_size)
        
        match color:
            case "invert":
                frame = cv.bitwise_not(frame)
            case "gray":
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            case "rainbow":
                frame = cv.applyColorMap(frame, cv.COLORMAP_RAINBOW)
            case "pink":
                frame = cv.applyColorMap(frame, cv.COLORMAP_PINK)
            case "colormap":
                frame = cv.applyColorMap(frame, cv.COLORMAP_JET)

        cv.imshow(name, frame)
        if cv.waitKey(15) & 0xFF == 27:
            break
    cv.destroyWindow(name)
    cap.release()

def task3():
    show_video("media/car.mp4", "Big Original video" , size="big")
    show_video("media/car.mp4", "Medium Gray", size="medium", color="gray")
    show_video("media/car.mp4", "Big Rainbow", size="big", color="rainbow")
    show_video("media/car.mp4", "Big Inverted Origin", size="big", color="invert")
    show_video("media/car.mp4", "Big Pink", size="big", color="pink")
    show_video("media/car.mp4", "Small Colormapped video", size="small", color="colormap")
task3()
    