from djitellopy import Tello
import numpy as np
import cv2

# Connect to Tello
drone = Tello()
drone.connect()
drone.streamoff()
drone.streamon()

width, height = 320, 240  # Resolution for processing/display

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
    img = cv2.resize(frame, (width, height))
    output = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "Control Panel")
    h_max = cv2.getTrackbarPos("Hue Max", "Control Panel")
    s_min = cv2.getTrackbarPos("Sat Min", "Control Panel")
    s_max = cv2.getTrackbarPos("Sat Max", "Control Panel")
    v_min = cv2.getTrackbarPos("Val Min", "Control Panel")
    v_max = cv2.getTrackbarPos("Val Max", "Control Panel")

    lower, upper = np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Draw video center
    center_video = (width // 2, height // 2)
    cv2.circle(output, center_video, 5, (0, 255, 0), -1)  # Green center dot

    # Detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 50,
                               param1=50, param2=30, minRadius=30, maxRadius=300)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Draw only the first detected circle
        for i in circles[0, :1]:
            x, y, r = i[0], i[1], i[2]
            center_circle = (x, y)

            # Draw circle outline and center dot
            cv2.circle(output, center_circle, r, (0, 255, 0), 2)
            cv2.circle(output, center_circle, 5, (0, 255, 0), -1)

            # Draw line connecting video center to circle center
            cv2.line(output, center_video, center_circle, (0, 255, 0), 1)

            # Compute and show distance
            dist = int(np.linalg.norm(np.array(center_circle) - np.array(center_video)))
            cv2.putText(output, f"Dist: {dist}px", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            break

    # Detect bullet holes
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Stack all views together
    top = np.hstack((img, result))
    bottom = np.hstack((cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), output))
    combined = np.vstack((top, bottom))

    cv2.imshow("Drone Vision Dashboard", combined)

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