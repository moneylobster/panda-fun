from pyvirtualdisplay import Display
import time

display=Display(visible=0, size=(1000,1000))
display.start()

from pynput.mouse import Button, Controller

mouse=Controller()

for i in range(5):
    print(f"Current mouse pos: {mouse.position}")
    time.sleep(1)
