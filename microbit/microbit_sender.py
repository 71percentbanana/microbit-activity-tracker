"""
MicroPython script for the BBC micro:bit.
Flashes onto the micro:bit to continuously read accelerometer X, Y, Z
in milli-g and stream them over USB serial at 20 Hz (50 ms intervals).

How to flash onto your micro:bit:
1. Open https://python.microbit.org/ in Chrome / Edge.
2. Paste this code.
3. Connect micro:bit via USB and click "Send to micro:bit" (or download the .hex file and drag-drop onto the MICROBIT drive).

MakeCode Blocks Alternative:
- In MakeCode (https://makecode.microbit.org/):
  Inside 'forever':
    serial write line (join [acceleration x], [","], [acceleration y], [","], [acceleration z])
    pause (50) ms
"""

from microbit import *

# Configure Serial/UART baud rate (115200 bps)
uart.init(baudrate=115200)

# Display a ready checkmark icon on the LED matrix
display.show(Image.YES)
sleep(500)
display.clear()

# 20 Hz loop (every 50 milliseconds)
while True:
    # Read 3-axis accelerometer values (in milli-g: -2000 to +2000 mg)
    x = accelerometer.get_x()
    y = accelerometer.get_y()
    z = accelerometer.get_z()

    # Transmit over USB serial
    uart.write(f"{x},{y},{z}\r\n")

    # Maintain ~20 Hz sampling rate
    sleep(50)
