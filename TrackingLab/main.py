import cv2 as cv
import numpy as np

lower1 = np.array([0, 120, 70])
upper1 = np.array([10, 255, 255])
lower2 = np.array([170, 120, 70])
upper2 = np.array([179, 255, 255])

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

cap = cv.VideoCapture(1, cv.CAP_AVFOUNDATION)  

while True:
    ok, img = cap.read()
    if not ok:
        break

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask1 = cv.inRange(hsv_img, lower1, upper1)
    mask2 = cv.inRange(hsv_img, lower2, upper2)
    mask = cv.bitwise_or(mask1, mask2)

    bw_filter = cv.dilate(cv.erode(mask, kernel, iterations=1), kernel, iterations=1)
    bw_filter = cv.erode(cv.dilate(bw_filter, kernel, iterations=1), kernel, iterations=1)

    contours, _ = cv.findContours(bw_filter, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    final_img = img.copy()

    if contours:  
        
        max_contour = max(contours, key=cv.contourArea)
        area = cv.contourArea(max_contour)

        if area > 500:
            x, y, w, h = cv.boundingRect(max_contour)
            M = cv.moments(max_contour)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                cv.circle(final_img, (cx, cy), 6, (0, 255, 0), -1)

            cv.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 0), 2)

            cv.putText(final_img, f"Area: {int(area)}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv.putText(final_img, f"Centroid: ({cx},{cy})", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv.putText(final_img, "No significant object", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    else:
        cv.putText(final_img, "No object detected", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv.imshow('HSV Image', hsv_img)
    cv.imshow('Find Red', final_img)
    cv.imshow('Red Mask', bw_filter)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()

