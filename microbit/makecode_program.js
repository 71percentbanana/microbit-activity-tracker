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
