import cv2
import numpy as np
import threading
import math
import time
from djitellopy import Tello

class TelloController:
    def __init__(self):
        self.tello = Tello()
        self.x = self.y = self.z = 0.0
        self.yaw = 0.0
        self.stream_on = False
        self.frame = None
        self._video_thread = None

    def connect(self):
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")
        self.tello.streamoff()
        self.tello.streamon()
        self.stream_on = True
        self._video_thread = threading.Thread(target=self._update_frame, daemon=True)
        self._video_thread.start()
        time.sleep(1)  # Quick warm-up

    def _update_frame(self):
        reader = self.tello.get_frame_read()
        while self.stream_on:
            try:
                raw = reader.frame
                self.frame = cv2.resize(raw, (320, 240))
            except Exception:
                pass
            time.sleep(1 / 60)  # High FPS loop

    def takeoff(self):
        self.tello.takeoff()
        self.z = 80.0
        print("Airborne!")

    def land(self):
        self.tello.land()
        self.z = 0.0
        print("Landed")

    def move(self, dx, dy, dz):
        rad = math.radians(self.yaw)
        forward = dx * math.cos(rad) + dy * math.sin(rad)
        right = -dx * math.sin(rad) + dy * math.cos(rad)

        if forward > 0:
            self.tello.move_forward(int(forward))
        elif forward < 0:
            self.tello.move_back(int(-forward))

        if right > 0:
            self.tello.move_right(int(right))
        elif right < 0:
            self.tello.move_left(int(-right))

        if dz > 0:
            self.tello.move_up(int(dz))
        elif dz < 0:
            self.tello.move_down(int(-dz))

        self.x += dx
        self.y += dy
        self.z += dz
        print(f"Moved to X:{self.x:.0f} Y:{self.y:.0f} Z:{self.z:.0f}")

    def rotate(self, deg):
        if deg > 0:
            self.tello.rotate_clockwise(int(deg))
        else:
            self.tello.rotate_counter_clockwise(int(-deg))
        self.yaw = (self.yaw + deg) % 360
        print(f"Yaw → {self.yaw:.0f}°")

    def cleanup(self):
        self.stream_on = False
        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join()
        try:
            self.tello.streamoff()
        except:
            pass
        cv2.destroyAllWindows()

def command_loop(drone: TelloController):
    print("\nCommands:\n takeoff | land | move x y z | rotate deg | exit\n")
    try:
        while drone.stream_on:
            cmd = input("> ").strip().lower()
            if not cmd:
                continue
            if cmd == "exit":
                if drone.z > 0: drone.land()
                drone.stream_on = False
                break
            elif cmd == "takeoff":
                drone.takeoff()
            elif cmd == "land":
                drone.land()
            elif cmd.startswith("move"):
                try:
                    _, x, y, z = cmd.split()
                    drone.move(float(x), float(y), float(z))
                except ValueError:
                    print("Usage: move x y z")
            elif cmd.startswith("rotate"):
                try:
                    _, deg = cmd.split()
                    drone.rotate(float(deg))
                except ValueError:
                    print("Usage: rotate deg")
            else:
                print("Invalid command")
    finally:
        drone.cleanup()

def main():
    drone = TelloController()
    drone.connect()

    # HSV control panel
    def empty(a): pass
    cv2.namedWindow("Control Panel")
    cv2.resizeWindow("Control Panel", 640, 240)
    cv2.createTrackbar("Hue Min", "Control Panel", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "Control Panel", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "Control Panel", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "Control Panel", 255, 255, empty)
    cv2.createTrackbar("Val Min", "Control Panel", 0, 255, empty)
    cv2.createTrackbar("Val Max", "Control Panel", 255, 255, empty)

    threading.Thread(target=command_loop, args=(drone,), daemon=True).start()

    while drone.stream_on:
        if drone.frame is None:
            continue

        img = drone.frame.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # HSV masking
        h_min = cv2.getTrackbarPos("Hue Min", "Control Panel")
        h_max = cv2.getTrackbarPos("Hue Max", "Control Panel")
        s_min = cv2.getTrackbarPos("Sat Min", "Control Panel")
        s_max = cv2.getTrackbarPos("Sat Max", "Control Panel")
        v_min = cv2.getTrackbarPos("Val Min", "Control Panel")
        v_max = cv2.getTrackbarPos("Val Max", "Control Panel")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        # Circle detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 50, param1=50, param2=30, minRadius=30, maxRadius=300)
        frame_center = (160, 120)

        if circles is not None:
            for i in np.uint16(np.around(circles))[0, :1]:
                center = (i[0], i[1])
                radius = i[2]
                cv2.circle(mask, center, radius, (255, 255, 255), 2)
                cv2.circle(mask, center, 3, (255, 255, 255), -1)
                cv2.circle(mask, frame_center, 3, (255, 255, 255), -1)
                cv2.line(mask, frame_center, center, (255, 255, 255), 1)
                dist = int(np.linalg.norm(np.array(center) - np.array(frame_center)))
                cv2.putText(mask, f"{dist}px", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                # Overlay on result
                cv2.circle(result, center, radius, (0, 255, 0), 2)
                cv2.circle(result, center, 3, (0, 255, 0), -1)
                cv2.circle(result, frame_center, 3, (0, 255, 0), -1)
                cv2.line(result, frame_center, center, (0, 255, 0), 1)
                cv2.putText(result, f"{dist}px", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Bullet holes
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 20 < cv2.contourArea(cnt) < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

        stack = np.vstack([
            np.hstack([img, result]),
            np.hstack([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), np.zeros_like(img)])
        ])
        cv2.imshow("Drone Vision Dashboard", stack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    drone.cleanup()

if __name__ == "__main__":
    main()