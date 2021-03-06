from pynput.mouse import Button, Controller
mouse = Controller()
print(mouse.position)
def init():
    global cX,cY
    cX = 0
    cY = 0