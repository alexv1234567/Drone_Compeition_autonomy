from djitellopy import Tello
import numpy as np
import cv2

# Connect to Tello
drone = Tello()
drone.connect()
drone.streamoff()
drone.streamon()

width, height = 320, 240

def empty(a): pass

# Create HSV controls
cv2.namedWindow("Control Panel")
cv2.resizeWindow("Control Panel", 640, 240)
cv2.createTrackbar("Hue Min", "Control Panel", 0, 179, empty)
cv2.createTrackbar("Hue Max", "Control Panel", 179, 179, empty)
cv2.createTrackbar("Sat Min", "Control Panel", 0, 255, empty)
cv2.createTrackbar("Sat Max", "Control Panel", 255, 255, empty)
cv2.createTrackbar("Val Min", "Control Panel", 0, 255, empty)
cv2.createTrackbar("Val Max", "Control Panel", 255, 255, empty)

while True:
    frame = drone.get_frame_read().frame
    frame = cv2.GaussianBlur(frame, (3, 3), 2)
    img = cv2.resize(frame, (width, height))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV filtering
    h_min = cv2.getTrackbarPos("Hue Min", "Control Panel")
    h_max = cv2.getTrackbarPos("Hue Max", "Control Panel")
    s_min = cv2.getTrackbarPos("Sat Min", "Control Panel")
    s_max = cv2.getTrackbarPos("Sat Max", "Control Panel")
    v_min = cv2.getTrackbarPos("Val Min", "Control Panel")
    v_max = cv2.getTrackbarPos("Val Max", "Control Panel")

    lower, upper = np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Circle detection
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.2, 50, param1=50, param2=30, minRadius=30, maxRadius=300)

    # Center of the image
    frame_center = (width // 2, height // 2)

    if circles is not None:
        for i in np.uint16(np.around(circles))[0, :1]:  # Take only the first circle
            circle_center = (i[0], i[1])
            radius = i[2]

            # Draw circle outline
            cv2.circle(result, circle_center, radius, (0, 255, 0), 2)

            # Green dot at center of circle
            cv2.circle(result, circle_center, 4, (0, 255, 0), -1)

            # Green dot at center of frame
            cv2.circle(result, frame_center, 4, (0, 255, 0), -1)

            # Draw line between circle center and frame center
            cv2.line(result, frame_center, circle_center, (0, 255, 0), 1)

            # Calculate and display pixel distance
            distance = int(np.linalg.norm(np.array(circle_center) - np.array(frame_center)))
            cv2.putText(result, f"Dist: {distance}px", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Optional: draw bullet hole bounding boxes
    _, thresh = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Compose GUI
    top_row = np.hstack((img, result))
    bottom_row = np.hstack((cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), np.zeros_like(img)))
    dashboard = np.vstack((top_row, bottom_row))

    cv2.imshow("Drone Vision Dashboard", dashboard)

# Print final HSV values
print("h_min:", h_min)
print("h_max:", h_max)
print("s_min:", s_min)
print("s_max:", s_max)
print("v_min:", v_min)
print("v_max:", v_max)

cv2.destroyAllWindows()