import struct

with open("/dev/input/mice", mode="rb") as f:
    while True:
        button, x, y = struct.unpack('BBB', f.read(3))
        # print('x={:08b}, y= {:08b}, button= {:08b}'.format(x, y, button))
        if x & 0b10000000 == 0b10000000:
            print(f"x left {x:>08b}")
        elif x & 0b00000001 == 0b00000001:
            print(f"x right {x:>08b}")
        else:
            print("idk x")
        if y & 0b10000000 == 0b10000000:
            print(f"y down {y:>08b}")
        elif y & 0b00000001 == 0b00000001:
            print(f"y up {y:>08b}")
        else:
            print("idk y")
        print("done")
