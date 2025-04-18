import cv2
import numpy as np
import threading
import time
from djitellopy import Tello

lower = np.array([105, 28, 94])
upper = np.array([179, 255, 160])

class TelloController:
    def __init__(self):
        self.tello = Tello()
        self.x = self.y = self.z = 0.0
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

    def manual(self):
        print("esc, w, a, s, d, e, f, u, d")
        print("exit, forward, left, back, right, cw, ccw, up, down")
        while True:
            key = cv2.waitKey(1) & 0xff
            if key == ord('w'):
                self.tello.move_forward(30)
            elif key == ord('s'):
                self.tello.move_back(30)
            elif key == ord('a'):
                self.tello.move_left(30)
            elif key == ord('d'):
                self.tello.move_right(30)
            elif key == ord('e'):
                self.tello.rotate_clockwise(30)
            elif key == ord('f'):
                self.tello.rotate_counter_clockwise(30)
            elif key == ord('u'):
                self.tello.move_up(30)
            elif key == ord('d'):
                self.tello.move_down(30)
            elif key == 27:
                break

    def land(self):
        self.tello.land()
        self.z = 0.0
        print("Landed")

    def move(self, dx, dy, dz):
        if dx > 0:
            self.tello.move_forward(dx)
        elif dx < 0:
            self.tello.move_back(dx)

        if dy > 0:
            self.tello.move_right(dy)
        elif dy < 0:
            self.tello.move_left(-dy)

        if dz > 0:
            self.tello.move_up(int(dz))
        elif dz < 0:
            self.tello.move_down(int(-dz))

        self.x += dx
        self.y += dy
        self.z += dz
        print(f"Moved to X:{self.x:.0f} Y:{self.y:.0f} Z:{self.z:.0f}")

    def detect_circle(frame):
        """Returns dx, dy, in_center, and px_to_cm (None if no circle found) dx and dy are in pixels relative to the center"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        frame = cv2.GaussianBlur(frame, (3, 3), 2)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        circles = cv2.HoughCircles(
            mask, cv2.HOUGH_GRADIENT, 1.2, 50,
            param1=50, param2=30,
            minRadius=20, maxRadius=200
        )

        if circles is not None:
            circle = np.uint16(np.around(circles))[0][0]
            cx, cy, radius = circle
            diameter_px = radius * 2

            # Calculate real-world cm per pixel
            px_to_cm = 17.78 / diameter_px

            # Define center 1/9th region bounds
            x1 = center_x - width // 6
            x2 = center_x + width // 6
            y1 = center_y - height // 6
            y2 = center_y + height // 6

            dx = cx - center_x
            dy = cy - center_y

            in_center = x1 <= cx <= x2 and y1 <= cy <= y2
            return dx, dy, in_center, px_to_cm, False
        elif circles is None:
            return 0, 0, False, None, True
        return 0, 0, False, None, False

    def accurate_landing(self, frame):
        print("Starting accurate landing...")
        while self.z > 30:
            key = cv2.waitKey(1) & 0xff
            if key == ord('d'):
                self.manual()

            frame = self.frame
            if frame is None:
                continue

            dx, dy, circle_detected, px_to_cm, orbit = self.detect_circle(frame)

            if circle_detected:
                print("Circle centered — descending 20cm")
                self.move(0, 0, -20)
                time.sleep(1)
            else:
                if orbit:
                    self.manual()
                else:
                    move_x = -dx * px_to_cm  # left/right
                    move_y = -dy * px_to_cm  # forward/backward
                    print(f"Adjusting position: dx={dx}px dy={dy}px → move({move_y:.1f}, {move_x:.1f}, 0)")
                    self.move(move_y, move_x, 0)
                    time.sleep(1)

        print("Hovering...")

    def part1(self, frame):
        print("Part 1")
        key = cv2.waitKey(1) & 0xff
        if key == ord('o'):
            self.manual()
        #Insert hard code directions to general location
        self.accurate_landing(self, frame)

    def part2(self, frame):
        print("Part 2")
        key = cv2.waitKey(1) & 0xff
        if key == ord('o'):
            self.manual()
        #Insert hard code directions to general location
        self.accurate_landing(self, frame)

    def part3(self, frame):
        print("Part 3")
        key = cv2.waitKey(1) & 0xff
        if key == ord('o'):
            self.manual()
        #Insert hard code directions to general location
        self.accurate_landing(self, frame)

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
    print("\nCommands:\n takeoff | land | 1, 2, 3 | move x y z | manual | exit\n")
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
                print("\nCommands:\n takeoff | land | 1, 2, 3 | move x y z | manual | exit\n")
            elif cmd == "land":
                drone.land()
                print("\nCommands:\n takeoff | land | 1, 2, 3 | move x y z | manual | exit\n")
            elif cmd == "1":
                drone.part1()
                print("\nCommands:\n takeoff | land | 1, 2, 3 | move x y z | manual | exit\n")
            elif cmd == "2":
                drone.part2
                print("\nCommands:\n takeoff | land | 1, 2, 3 | move x y z | manual | exit\n")
            elif cmd == "manual":
                drone.manual()
                print("\nCommands:\n takeoff | land | 1, 2, 3 | move x y z | manual | exit\n")
            elif cmd.startswith("move"):
                try:
                    _, x, y, z = cmd.split()
                    drone.move(float(x), float(y), float(z))
                except ValueError:
                    print("Usage: move x y z")
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
    cv2.createTrackbar("Hue Min", "Control Panel", 118, 179, empty)
    cv2.createTrackbar("Hue Max", "Control Panel", 141, 179, empty)
    cv2.createTrackbar("Sat Min", "Control Panel", 17, 255, empty)
    cv2.createTrackbar("Sat Max", "Control Panel", 78, 255, empty)
    cv2.createTrackbar("Val Min", "Control Panel", 145, 255, empty)
    cv2.createTrackbar("Val Max", "Control Panel", 254, 255, empty)

    threading.Thread(target=command_loop, args=(drone,), daemon=True).start()

    while drone.stream_on:
        if drone.frame is None:
            continue

        frame = drone.frame.copy()

        # HSV masking
        h_min = cv2.getTrackbarPos("Hue Min", "Control Panel")
        h_max = cv2.getTrackbarPos("Hue Max", "Control Panel")
        s_min = cv2.getTrackbarPos("Sat Min", "Control Panel")
        s_max = cv2.getTrackbarPos("Sat Max", "Control Panel")
        v_min = cv2.getTrackbarPos("Val Min", "Control Panel")
        v_max = cv2.getTrackbarPos("Val Max", "Control Panel")
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Circle detection
        img = cv2.GaussianBlur(frame, (3, 3), 2)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)
        masked_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(masked_gray, (7, 7), 2)
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

        #test
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

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