from pynput.mouse import Button, Controller
import time

mouse=Controller()

for i in range(5):
    print(f"Current mouse pos: {mouse.position}")
    time.sleep(1)
