# ESP32-S3 Cam Rover (XIAO ESP32S3 Sense)

A tiny rover that streams 240×240 MJPEG from the onboard OV2640 and drives two FS90R continuous rotation servos from a web UI (buttons + arrow keys). The sketch keeps latency low by using double camera buffers, `CAMERA_GRAB_LATEST`, and Wi-Fi sleep disabled.

## Hardware
- Board: Seeed Studio XIAO ESP32S3 Sense (built-in OV2640).
- Motors: 2× FS90R continuous servo wheels.
- Power: Servos prefer 5 V with adequate current (USB 5 V rail works for light load; a separate 5 V pack is safer). Always tie grounds together.

### Pinout used
| Function | GPIO | XIAO silk | Notes |
| -------- | ---- | --------- | ----- |
| Left servo signal | GPIO2 | D1 | PWM via LEDC channel 4 |
| Right servo signal | GPIO4 | D3 | PWM via LEDC channel 5 |
| Servo power | 5 V | 5V | Shared with servos (or external 5 V) |
| Servo ground | GND | GND | Must share ground with ESP32-S3 |
| Camera | Onboard | — | Pins defined in `camera_pins.h` |

If a wheel spins the wrong way, flip the sign constants `LEFT_DIR` or `RIGHT_DIR` in `webcam_car_controller.ino`.

## Build & upload (Arduino IDE)
1) Install the ESP32 board package and select **Seeed XIAO ESP32S3 Sense**.  
2) Open `webcam_car_controller/webcam_car_controller.ino`.  
3) Set your Wi‑Fi credentials at the top of the sketch.  
4) Compile & upload. Open Serial Monitor (115200) to note the IP.

## Use
- Open `http://<device-ip>/` for the control UI. Stream comes from port `81` at `http://<device-ip>:81/stream`.
- Controls: hold on-screen arrows or use keyboard **↑ ↓ ← →**; release stops both wheels. Center square is a stop button.
- API: `GET /drive?l=<-1..1>&r=<-1..1>` sets left/right speeds (‑1 reverse, 0 stop, 1 forward). `GET /status` returns current direction settings.

## Tuning & latency tips
- Resolution is fixed to `FRAMESIZE_240X240`; quality set to `s->set_quality(...12)` with `jpeg_quality 14`. Increase the quality number (e.g., 16–20) if bandwidth is tight; decrease for crisper frames.
- Double buffering (`fb_count=2`) plus `CAMERA_GRAB_LATEST` reduces lag.
- `WiFi.setSleep(false)` keeps radio awake for faster response.
- Adjust `cmds` map in the embedded HTML to change turn aggressiveness.

## Safety notes
- Servos draw spikes; if video drops while driving, move servos to a dedicated 5 V supply and common ground.
- FS90R uses pulse width: ~1500 µs stop, shorter reverse, longer forward. Values are clamped 1000–2000 µs in the code. Adjust `SERVO_MIN_US`, `SERVO_MAX_US`, or `SERVO_STOP_US` if your units differ.
