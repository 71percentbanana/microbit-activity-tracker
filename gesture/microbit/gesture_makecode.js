// Microsoft MakeCode JavaScript for BBC micro:bit - Gesture Classifier & Activity Tracker
// Streams 3-axis accelerometer readings over USB Serial (~20-25 Hz in practice) and shows the
// gesture or activity the laptop recognised on the LED matrix (see ../../dashboard).
//
// Controls:
//   Hold button A while performing a gesture - the centre LED lights while recording.
//   Set FLIP_DISPLAY below to false if you hold the board the normal way round.
//   Button B clears the display.
//
// Serial protocol (micro:bit -> laptop), one line each:
//   x,y,z      accelerometer reading in milli-g, sent continuously
//   START      button A pressed: the gesture begins
//   END        button A released: the gesture ends
//
// Serial protocol (laptop -> micro:bit), one line each:
//   LEFT / RIGHT / UP / DOWN / PUSH / PULL /
//   CIRCLE / SHAKE / TILT_LEFT / TILT_RIGHT /
//   TAP / DOUBLE_TAP                            recognised gesture
//   1 ... 9                                     recognised air-written number
//   CUSTOM                                      recognised custom gesture (shows a star)
//   ?                                           not confident
//   SAVED                                       recording saved (Record mode)
//   ACT:WALKING / ACT:JOGGING / ACT:UPSTAIRS /
//   ACT:DOWNSTAIRS / ACT:STANDING / ACT:SITTING  current activity (Activity view)
//
// Instructions:
// 1. Open https://makecode.microbit.org/
// 2. Click "New Project"
// 3. Switch top tab from "Blocks" to "JavaScript"
// 4. Paste this code and click "Download" to flash your micro:bit

// Set to true when the board is held turned around (rotated 180°), so everything
// on the LEDs - arrows, icons and numbers - appears the right way up to the user
const FLIP_DISPLAY = true

let recording = false
let reply = ""

// Shows an image, rotated 180° when FLIP_DISPLAY is set
function show(img: Image) {
    if (!FLIP_DISPLAY) {
        img.showImage(0, 0)
        return
    }
    let flipped = images.createImage(`
        . . . . .
        . . . . .
        . . . . .
        . . . . .
        . . . . .
        `)
    for (let x = 0; x < 5; x++) {
        for (let y = 0; y < 5; y++) {
            flipped.setPixel(4 - x, 4 - y, img.pixel(x, y))
        }
    }
    flipped.showImage(0, 0)
}

// 5x5 characters for numbers, ? and activity letters (drawn with show() so they can be flipped)
function charImage(c: string): Image {
    if (c == "1") {
        return images.createImage(`
            . . # . .
            . # # . .
            . . # . .
            . . # . .
            . # # # .
            `)
    } else if (c == "2") {
        return images.createImage(`
            # # # . .
            . . . # .
            . # # . .
            # . . . .
            # # # # .
            `)
    } else if (c == "3") {
        return images.createImage(`
            # # # # .
            . . . # .
            . . # . .
            # . . # .
            . # # . .
            `)
    } else if (c == "4") {
        return images.createImage(`
            . . # # .
            . # . # .
            # . . # .
            # # # # #
            . . . # .
            `)
    } else if (c == "5") {
        return images.createImage(`
            # # # # #
            # . . . .
            # # # # .
            . . . . #
            # # # # .
            `)
    } else if (c == "6") {
        return images.createImage(`
            . . . # .
            . . # . .
            . # # # .
            # . . . #
            . # # # .
            `)
    } else if (c == "7") {
        return images.createImage(`
            # # # # #
            . . . # .
            . . # . .
            . # . . .
            # . . . .
            `)
    } else if (c == "8") {
        return images.createImage(`
            . # # # .
            # . . . #
            . # # # .
            # . . . #
            . # # # .
            `)
    } else if (c == "9") {
        return images.createImage(`
            . # # # .
            # . . . #
            . # # # .
            . . # . .
            . # . . .
            `)
    } else if (c == "W") {
        return images.createImage(`
            # . . . #
            # . . . #
            # . # . #
            # # . # #
            # . . . #
            `)
    } else if (c == "J") {
        return images.createImage(`
            # # # # #
            . . . # .
            . . . # .
            # . . # .
            . # # . .
            `)
    } else if (c == "R") {
        return images.createImage(`
            # # # # .
            # . . . #
            # # # # .
            # . . # .
            # . . . #
            `)
    } else if (c == "E") {
        return images.createImage(`
            # # # # #
            # . . . .
            # # # # .
            # . . . .
            # # # # #
            `)
    } else if (c == "A") {
        return images.createImage(`
            . # # # .
            # . . . #
            # # # # #
            # . . . #
            # . . . #
            `)
    } else if (c == "D") {
        return images.createImage(`
            # # # # .
            # . . . #
            # . . . #
            # . . . #
            # # # # .
            `)
    } else if (c == "Y") {
        return images.createImage(`
            # . . . #
            . # . # .
            . . # . .
            . . # . .
            . . # . .
            `)
    }
    // "?"
    return images.createImage(`
        . # # # .
        # . . . #
        . . # # .
        . . . . .
        . . # . .
        `)
}

serial.redirectToUSB()
serial.setBaudRate(BaudRate.BaudRate115200)

// Show READY one letter at a time, so it can be flipped too
for (let letter of ["R", "E", "A", "D", "Y"]) {
    show(charImage(letter))
    basic.pause(350)
}
basic.clearScreen()

// Show the gesture or activity sent back by the laptop
serial.onDataReceived(serial.delimiters(Delimiters.NewLine), function () {
    reply = serial.readUntil(serial.delimiters(Delimiters.NewLine))
    if (reply == "LEFT") {
        show(images.arrowImage(ArrowNames.West))
    } else if (reply == "RIGHT") {
        show(images.arrowImage(ArrowNames.East))
    } else if (reply == "UP") {
        show(images.arrowImage(ArrowNames.North))
    } else if (reply == "DOWN") {
        show(images.arrowImage(ArrowNames.South))
    } else if (reply == "SHAKE") {
        show(images.iconImage(IconNames.Chessboard))
    } else if (reply == "CIRCLE") {
        show(images.createImage(`
            . # # # .
            # . . . #
            # . . . #
            # . . . #
            . # # # .
            `))
    } else if (reply == "PUSH") {
        // Moving away: small square
        show(images.iconImage(IconNames.SmallSquare))
    } else if (reply == "PULL") {
        // Coming closer: big square
        show(images.iconImage(IconNames.Square))
    } else if (reply == "TILT_LEFT") {
        show(images.arrowImage(ArrowNames.SouthWest))
    } else if (reply == "TILT_RIGHT") {
        show(images.arrowImage(ArrowNames.SouthEast))
    } else if (reply == "TAP") {
        show(images.createImage(`
            . . . . .
            . . # . .
            . . # . .
            . . # . .
            . . . . .
            `))
    } else if (reply == "DOUBLE_TAP") {
        show(images.createImage(`
            . . . . .
            . # . # .
            . # . # .
            . # . # .
            . . . . .
            `))
    } else if (reply == "CUSTOM") {
        // A custom gesture added from the dashboard: star
        show(images.createImage(`
            . . # . .
            # # # # #
            . # # # .
            . # . # .
            # . . . #
            `))
    } else if (reply == "SAVED") {
        show(images.iconImage(IconNames.Yes))
    } else if (reply.length == 1) {
        // A number 1-9, or ? when not confident
        show(charImage(reply))
    } else if (reply == "ACT:WALKING") {
        show(charImage("W"))
    } else if (reply == "ACT:JOGGING") {
        show(charImage("J"))
    } else if (reply == "ACT:UPSTAIRS") {
        show(images.arrowImage(ArrowNames.NorthEast))
    } else if (reply == "ACT:DOWNSTAIRS") {
        show(images.arrowImage(ArrowNames.SouthWest))
    } else if (reply == "ACT:STANDING") {
        show(images.createImage(`
            . . # . .
            . # # # .
            . . # . .
            . # . # .
            . # . # .
            `))
    } else if (reply == "ACT:SITTING") {
        show(images.createImage(`
            . # . . .
            . # . . .
            . # # # .
            . # . # .
            . # . # .
            `))
    }
})

input.onButtonPressed(Button.B, function () {
    basic.clearScreen()
})

// Main sampling loop (pause 20 ms; ~20-25 readings per second in practice)
basic.forever(function () {
    if (input.buttonIsPressed(Button.A)) {
        if (!recording) {
            recording = true
            basic.clearScreen()
            led.plot(2, 2)
            serial.writeLine("START")
        }
    } else if (recording) {
        recording = false
        led.unplot(2, 2)
        serial.writeLine("END")
    }

    // Sends comma-separated values: "x,y,z"
    serial.writeLine("" + input.acceleration(Dimension.X) + "," + input.acceleration(Dimension.Y) + "," + input.acceleration(Dimension.Z))

    // 20 ms pause between readings
    basic.pause(20)
})
