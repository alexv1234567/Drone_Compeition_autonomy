from djitellopy import Tello
import numpy as np
import cv2

drone = Tello()
drone.connect()
drone.streamoff()
drone.streamon()

width = 640
height = 400
cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

def empty(a):
    pass

cv2.nameWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("Hue Min", "HSV", 0, 179, empty)
cv2.createTrackbar("Hue Max", "HSV", 0, 179, empty)
cv2.createTrackbar("Sat Min", "HSV", 0, 179, empty)
cv2.createTrackbar("Sat Max", "HSV", 0, 179, empty)
cv2.createTrackbar("Val Min", "HSV", 0, 179, empty)
cv2.createTrackbar("Val Max", "HSV", 0, 179, empty)

while True:
    print(drone.get_battery())
    frame_read = drone.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "HSV")
    h_max = cv2.getTrackbarPos("Hue Max", "HSV")
    s_min = cv2.getTrackbarPos("Sat Min", "HSV")
    s_max = cv2.getTrackbarPos("Sat Max", "HSV")
    v_min = cv2.getTrackbarPos("Val Min", "HSV")
    v_max = cv2.getTrackbarPos("Val Min", "HSV")

    lower = np.array(([h_min, s_min, v_min]))
    upper = np.array(([h_max, s_max, v_max]))
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask = mask)

    cv2.imshow(img)
    cv2.imshow(imgHsv)
    cv2.imshow(mask)
    cv2.imshow(result)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

print("h_min: ", h_min)
print("h_max: ", h_max)
print("s_min: ", s_min)
print("s_max: ", s_max)
print("v_min: ", v_min)
print("v_max: ", h_max)