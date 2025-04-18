#!/usr/bin/env python3
"""
Mac‑friendly Tello controller with live video HUD
=================================================
• All OpenCV GUI calls run on the **main thread** (required on macOS).
• Text commands run in a background thread, so you can keep typing while
  the window is open.

Commands
--------
takeoff           – Launch drone
land              – Land drone
move x y z        – Translate relative to current pose (cm)
rotate deg        – Rotate about the Z‑axis (degrees, +CW / –CCW)
exit              – Quit (auto‑lands if airborne)
Press “q” in the video window to quit as well.
"""

from djitellopy import Tello
import cv2
import numpy as np
import threading
import math
import time


class TelloController:
    def __init__(self) -> None:
        self.tello = Tello()

        # Estimated global pose (cm / deg)
        self.x = self.y = self.z = 0.0
        self.yaw = 0.0

        # Video‑streaming state
        self.stream_on = False
        self.frame = None
        self._video_thread: threading.Thread | None = None

    # --------------------------------------------------------------------- #
    # Drone connection & video capture                                      #
    # --------------------------------------------------------------------- #
    def connect(self) -> None:
        """Connect to the drone and start the video stream."""
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")

        self.tello.streamon()
        self.stream_on = True

        # Background thread for frame capture
        self._video_thread = threading.Thread(
            target=self._update_frame, daemon=True
        )
        self._video_thread.start()

        time.sleep(2)  # give the camera a moment to warm up

    def _update_frame(self) -> None:
        """Continuously grab frames into `self.frame`."""
        frame_reader = self.tello.get_frame_read()
        while self.stream_on:
            try:
                raw = frame_reader.frame  # BGR
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                self.frame = cv2.resize(rgb, (640, 480))
            except Exception:
                # swallow occasional decode errors
                pass
            time.sleep(1 / 30)  # ≈30 FPS

    # --------------------------------------------------------------------- #
    # Flight commands                                                       #
    # --------------------------------------------------------------------- #
    def takeoff(self) -> None:
        self.tello.takeoff()
        self.z = 80.0
        print("Airborne!")

    def land(self) -> None:
        self.tello.land()
        self.z = 0.0
        print("Landed")

    def move(self, dx: float, dy: float, dz: float) -> None:
        """
        Translate relative to current pose (cm).

        dx, dy — forward/right in drone body frame
        dz     — positive up
        """
        rad = math.radians(self.yaw)
        forward = dx * math.cos(rad) + dy * math.sin(rad)
        right = -dx * math.sin(rad) + dy * math.cos(rad)

        if forward > 0:
            self.tello.move_forward(int(abs(forward)))
        elif forward < 0:
            self.tello.move_back(int(abs(forward)))

        if right > 0:
            self.tello.move_right(int(abs(right)))
        elif right < 0:
            self.tello.move_left(int(abs(right)))

        if dz > 0:
            self.tello.move_up(int(abs(dz)))
        elif dz < 0:
            self.tello.move_down(int(abs(dz)))

        self.x += dx
        self.y += dy
        self.z += dz

        print(f"Position → X={self.x:.0f}  Y={self.y:.0f}  Z={self.z:.0f}")

    def rotate(self, deg: float) -> None:
        if deg > 0:
            self.tello.rotate_clockwise(int(deg))
        else:
            self.tello.rotate_counter_clockwise(int(abs(deg)))
        self.yaw = (self.yaw + deg) % 360
        print(f"Yaw → {self.yaw:.0f}°")

    # --------------------------------------------------------------------- #
    # Video display                                                         #
    # --------------------------------------------------------------------- #
    def show_video(self) -> None:
        """Run in the **main** thread — displays the live video HUD."""
        cv2.namedWindow("Tello Front Camera")

        while self.stream_on:
            if self.frame is not None:
                hud = cv2.cvtColor(self.frame.copy(), cv2.COLOR_RGB2BGR) # RGB
                cv2.putText(
                    hud,
                    f"Pos  X:{self.x:.0f}  Y:{self.y:.0f}  Z:{self.z:.0f} cm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    hud,
                    f"Yaw: {self.yaw:.0f}°",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    hud,
                    f"Battery: {self.tello.get_battery()}%",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Tello Front Camera", cv2.cvtColor(hud, cv2.COLOR_RGB2BGR))

            # Quit video on “q”
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stream_on = False

        cv2.destroyAllWindows()

    # --------------------------------------------------------------------- #
    # Cleanup                                                               #
    # --------------------------------------------------------------------- #
    def cleanup(self) -> None:
        """Safely shut down streaming and video thread."""
        self.stream_on = False
        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join()

        try:
            self.tello.streamoff()
        except Exception:
            pass
        cv2.destroyAllWindows()


# ------------------------------------------------------------------------- #
# Background command‑line interface                                         #
# ------------------------------------------------------------------------- #
def command_loop(drone: TelloController) -> None:
    """
    Runs in its own thread so the main thread can manage the OpenCV window.
    """
    print(
        "\nTello Controller with Front Camera\n"
        "Commands:\n"
        "  takeoff       - Launch drone\n"
        "  land          - Land drone\n"
        "  move x y z    - Move (e.g., 'move 50 -30 0')\n"
        "  rotate deg    - Rotate (e.g., 'rotate -90')\n"
        "  exit          - Quit (auto‑lands if airborne)\n\n"
        "Press 'q' in the video window to stop streaming\n"
    )

    try:
        while drone.stream_on:
            cmd = input("> ").strip().lower()
            if not cmd:
                continue

            if cmd == "exit":
                if drone.z > 0:
                    drone.land()
                drone.stream_on = False
                break

            if cmd == "takeoff":
                drone.takeoff()
            elif cmd == "land":
                drone.land()
            elif cmd.startswith("move"):
                try:
                    _, sx, sy, sz = cmd.split()
                    drone.move(float(sx), float(sy), float(sz))
                except ValueError:
                    print("Usage: move x y z  (e.g.,  move 50 -20 10)")
            elif cmd.startswith("rotate"):
                try:
                    _, sdeg = cmd.split()
                    drone.rotate(float(sdeg))
                except ValueError:
                    print("Usage: rotate deg  (e.g.,  rotate -45)")
            else:
                print("Invalid command")
    finally:
        drone.cleanup()


# ------------------------------------------------------------------------- #
# Entry point                                                               #
# ------------------------------------------------------------------------- #
def main() -> None:
    drone = TelloController()
    drone.connect()

    # Start the CLI thread
    threading.Thread(target=command_loop, args=(drone,), daemon=True).start()

    # GUI loop (must remain on the main thread)
    drone.show_video()


if __name__ == "__main__":
    main()
