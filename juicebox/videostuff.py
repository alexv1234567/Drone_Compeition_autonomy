from djitellopy import Tello
import numpy as np
import cv2

# Connect to drone
drone = Tello()
drone.connect()
drone.streamoff()
drone.streamon()

# Set frame dimensions
width = 640
height = 400

# Dummy callback for trackbars
def empty(a):
    pass

# Create HSV trackbar window
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("Hue Min", "HSV", 0, 179, empty)
cv2.createTrackbar("Hue Max", "HSV", 179, 179, empty)
cv2.createTrackbar("Sat Min", "HSV", 0, 255, empty)
cv2.createTrackbar("Sat Max", "HSV", 255, 255, empty)
cv2.createTrackbar("Val Min", "HSV", 0, 255, empty)
cv2.createTrackbar("Val Max", "HSV", 255, 255, empty)

while True:
    print("Battery:", drone.get_battery())
    frame_read = drone.get_frame_read()
    img = frame_read.frame
    img = cv2.resize(img, (width, height))
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "HSV")
    h_max = cv2.getTrackbarPos("Hue Max", "HSV")
    s_min = cv2.getTrackbarPos("Sat Min", "HSV")
    s_max = cv2.getTrackbarPos("Sat Max", "HSV")
    v_min = cv2.getTrackbarPos("Val Min", "HSV")
    v_max = cv2.getTrackbarPos("Val Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show all the images
    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHsv)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

# Print final HSV values
print("h_min:", h_min)
print("h_max:", h_max)
print("s_min:", s_min)
print("s_max:", s_max)
print("v_min:", v_min)
print("v_max:", v_max)

cv2.destroyAllWindows()