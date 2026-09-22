// Microsoft MakeCode JavaScript for BBC micro:bit
// Streams 3-axis accelerometer readings over USB Serial at 20 Hz (every 50 ms)
// 
// Instructions:
// 1. Open https://makecode.microbit.org/
// 2. Click "New Project"
// 3. Switch top tab from "Blocks" to "JavaScript"
// 4. Paste this code and click "Download" to flash your micro:bit

// Configure USB serial at 115200 baud
serial.redirectToUSB()
serial.setBaudRate(BaudRate.BaudRate115200)

// Show a checkmark to indicate it is ready
basic.showIcon(IconNames.Yes)
basic.pause(500)
basic.clearScreen()

// Main sampling loop (20 Hz)
basic.forever(function () {
    let x = input.acceleration(Dimension.X)
    let y = input.acceleration(Dimension.Y)
    let z = input.acceleration(Dimension.Z)

    // Sends comma-separated values: "x,y,z"
    serial.writeLine(x + "," + y + "," + z)

    // 50 ms pause = 20 Hz
    basic.pause(50)
})
