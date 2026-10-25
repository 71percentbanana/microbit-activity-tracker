// Microsoft MakeCode JavaScript for BBC micro:bit
// Records 3-axis accelerometer readings and streams them over USB Serial at 20 Hz (every 50 ms)
//
// Controls:
//   Button A - start recording (shows "REC", resets activity statistics)
//   Button B - stop recording (shows a checkmark)
//
// Serial line format (one line per sample):
//   sampleCount,x,y,z,strength,activityLevel,averageActivity,peakActivity
//   x, y, z and strength are in milli-g; activityLevel = |strength - 1000|
//
// Instructions:
// 1. Open https://makecode.microbit.org/
// 2. Click "New Project"
// 3. Switch top tab from "Blocks" to "JavaScript"
// 4. Paste this code and click "Download" to flash your micro:bit

input.onButtonPressed(Button.A, function () {
    isRecording = true
    sampleCount = 0
    totalActivity = 0
    peakActivity = 0
    basic.showString("REC")
})
input.onButtonPressed(Button.B, function () {
    isRecording = false
    basic.showIcon(IconNames.Yes)
})
let averageActivity = 0
let activityLevel = 0
let strength = 0
let z = 0
let y = 0
let x = 0
let peakActivity = 0
let totalActivity = 0
let sampleCount = 0
let isRecording = false
serial.redirectToUSB()
basic.showString("READY")
basic.forever(function () {
    if (isRecording) {
        x = input.acceleration(Dimension.X)
        y = input.acceleration(Dimension.Y)
        z = input.acceleration(Dimension.Z)
        strength = input.acceleration(Dimension.Strength)
        activityLevel = Math.abs(strength - 1000)
        totalActivity += activityLevel
        sampleCount += 1
        averageActivity = totalActivity / sampleCount
        if (activityLevel > peakActivity) {
            peakActivity = activityLevel
        }
        serial.writeLine("" + sampleCount + "," + x + "," + y + "," + z + "," + strength + "," + activityLevel + "," + averageActivity + "," + peakActivity)
        // 50 ms pause = 20 Hz, matching the model's training sampling rate
        basic.pause(50)
    }
})
