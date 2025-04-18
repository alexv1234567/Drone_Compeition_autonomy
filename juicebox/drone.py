from djitellopy import Tello
import math
import cv2
from threading import Thread
import timeit

class TelloController:
    def __init__(self):
        self.tello = Tello()
        self.x, self.y, self.z = 0, 0, 0  # Global position (cm)
        self.yaw = 0  # Current rotation (degrees)
        
    def connect(self):
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")

    def takeoff(self):
        self.tello.takeoff()
        self.z = 80  # Takeoff height
        print("Airborne!")

    def land(self):
        self.tello.land()
        self.z = 0
        print("Landed")

    def move(self, dx, dy, dz):
        """Move relative to current position (supports negatives)"""
        # Convert global direction to drone-oriented movement
        rad = math.radians(self.yaw)
        forward = dx * math.cos(rad) + dy * math.sin(rad)
        right = -dx * math.sin(rad) + dy * math.cos(rad)
        
        # Execute movements (negative values move backward/left/down)
        if forward > 0:
            self.tello.move_forward(abs(forward))
        elif forward < 0:
            self.tello.move_back(abs(forward))
            
        if right > 0:
            self.tello.move_right(abs(right))
        elif right < 0:
            self.tello.move_left(abs(right))
            
        if dz > 0:
            self.tello.move_up(abs(dz))
        elif dz < 0:
            self.tello.move_down(abs(dz))
        
        # Update global position
        self.x += dx
        self.y += dy
        self.z += dz
        print(f"Position: X={self.x}, Y={self.y}, Z={self.z}")

    def rotate(self, degrees):
        """Rotate clockwise (negative for counter-clockwise)"""
        if degrees > 0:
            self.tello.rotate_clockwise(degrees)
        else:
            self.tello.rotate_counter_clockwise(abs(degrees))
        self.yaw = (self.yaw + degrees) % 360
        print(f"Yaw: {self.yaw}°")

#def videoRecorder(t, video, frame_read):
    #if t % 30 == 0:
       # video.write(frame_read.frame)

def main():
    drone = TelloController()
    drone.connect()
    #t_init = timeit.default_timer()
    
    print("\nTello Advanced Controller")
    print("Commands:")
    print("  takeoff       - Launch drone")
    print("  land          - Land drone")
    print("  move x y z    - Move (e.g., 'move 50 -30 0' for 50cm forward, 30cm left)")
    print("  rotate deg    - Rotate (e.g., 'rotate -90' for 90° counter-clockwise)")
    print("  exit          - Quit")

    try:
        drone.streamon()
        #video = cv2.VideoWriter('test.mp4', cv2.VdeoWriter_fourcc(*'MP4V'), 30, (1080, 720))
        #recorder = Thread(target=videoRecorder)
        #recorder.start()
    except:
        print("no video :(")

    while True:
        try:
            frame = drone.get_frame_read().frame
            #videoRecorder(timeit.default_timer - t_init,video,drone,frame)
            cv2.imshow("", frame)
        except:
            print("")

        cmd = input("> ").strip().lower()
        
        if cmd == "exit":
            if drone.z > 0: drone.land()
            break
        elif cmd == "takeoff":
            drone.takeoff()
        elif cmd == "land":
            recorder.join()
            drone.land()
        elif cmd.startswith("move"):
            try:
                _, x, y, z = cmd.split()
                drone.move(float(x), float(y), float(z))
            except:
                print("Usage: move x y z (e.g., 'move 50 -20 10')")
        elif cmd.startswith("rotate"):
            try:
                _, deg = cmd.split()
                drone.rotate(int(deg))
            except:
                print("Usage: rotate deg (e.g., 'rotate -45')")
        else:
            print("Invalid command")

if __name__ == "__main__":
    main()