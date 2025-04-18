import cv2
import numpy as np
import threading
import time
from djitellopy import Tello

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

    def detect_circle(frame, color):
        """Returns dx, dy, in_center, and px_to_cm (None if no circle found) dx and dy are in pixels relative to the center"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        if color == 1:
            lower = np.array([109, 28, 94])
            upper = np.array([179, 255, 160])
        #if color == 2:
            #set lower hsv boundaries
            #set upper hsv boundaries

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1.2, 50,
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
            return dx, dy, in_center, px_to_cm

        return 0, 0, False, None

    def accurate_landing(self, frame, color):
        print("Starting accurate landing...")
        while self.z > 30:
            frame = self.frame
            if frame is None:
                continue

            dx, dy, circle_detected, px_to_cm = self.detect_circle(frame, color)

            if circle_detected:
                print("Circle centered — descending 20cm")
                self.move(0, 0, -20)
                time.sleep(1)
            else:
                move_x = -dx * px_to_cm  # left/right
                move_y = -dy * px_to_cm  # forward/backward
                print(f"Adjusting position: dx={dx}px dy={dy}px → move({move_y:.1f}, {move_x:.1f}, 0)")
                self.move(move_y, move_x, 0)
                time.sleep(1)

        print("Landing...")
        self.land()

    def part1(self, frame):
        print("Part 1")
        #Hard code direction to general location
        self.accurate_landing(self, frame, 0)

    def part2(self, frame):
        print("Part 2")
        #Hard code direction to general location
        self.accurate_landing(self, frame, 1)

    def part3(self, frame):
        print("Part 3")
        #Hard code direction to general location
        self.accurate_landing(self, frame, 0)

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
        masked_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # Only color-masked regions
        blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)
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