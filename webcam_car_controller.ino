#include <Arduino.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "img_converters.h"
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <WiFi.h>
#include "esp32-hal-ledc.h"
#include "driver/ledc.h"

// Camera model used in the existing project
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

// ===========================
// Wi-Fi credentials (update)
// ===========================
const char *ssid = "BELL446";
const char *password = "iEEE2023?";

// ================
// Servo definitions
// ================
// FS90R continuous servos: center pulse (~1500µs) = stop, shorter = reverse, longer = forward.
// Pins chosen to avoid camera pins on the XIAO ESP32S3.
const int LEFT_SERVO_PIN = 2;   // D1 on XIAO header
const int RIGHT_SERVO_PIN = 4;  // D3 on XIAO header
// Invert if a wheel spins the opposite direction you expect
const int LEFT_DIR = 1;   // set to -1 to flip
const int RIGHT_DIR = 1;  // set to -1 to flip
// Per-servo trim in microseconds to nail the true stop point of your units
// Increase a trim value if that wheel creeps forward, decrease if it creeps backward.
const int LEFT_TRIM_US = 0;
const int RIGHT_TRIM_US = 0;
const float SERVO_DEADBAND = 0.05f;  // ignore tiny commands within ±5%

const ledc_mode_t SERVO_SPEED_MODE = LEDC_LOW_SPEED_MODE;
const ledc_timer_t SERVO_TIMER = LEDC_TIMER_1;
const ledc_channel_t LEFT_SERVO_CHANNEL = LEDC_CHANNEL_1;
const ledc_channel_t RIGHT_SERVO_CHANNEL = LEDC_CHANNEL_2;
const int SERVO_FREQUENCY = 50;          // 50 Hz standard servo PWM
const int SERVO_TIMER_RES_BITS = 14;     // resolution (bits) used in duty math
const ledc_timer_bit_t SERVO_RES_BITS = LEDC_TIMER_14_BIT;  // 14-bit for compatibility
const int SERVO_MIN_US = 1000;           // reverse
const int SERVO_STOP_US = 1500;          // stop / idle
const int SERVO_MAX_US = 2000;           // forward

// =====================
// Streaming definitions
// =====================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %ld.%06ld\r\n\r\n";

static httpd_handle_t camera_httpd = NULL;
static httpd_handle_t stream_httpd = NULL;
static uint8_t *last_jpg_buf = NULL;
static size_t last_jpg_size = 0;
static size_t last_jpg_cap = 0;
static bool last_jpg_valid = false;

// =================
// Utility functions
// =================
uint32_t usToDuty(int microseconds) {
  const uint32_t max_duty = (1u << SERVO_TIMER_RES_BITS) - 1;
  // duty cycle fraction = pulse_width / period. period = 1 / SERVO_FREQUENCY
  return (uint32_t)(((uint64_t)microseconds * max_duty * SERVO_FREQUENCY) / 1000000ULL);
}

void writeServoUs(uint8_t channel, int microseconds) {
  microseconds = constrain(microseconds, SERVO_MIN_US, SERVO_MAX_US);
  uint32_t duty = usToDuty(microseconds);
  ledc_set_duty(SERVO_SPEED_MODE, (ledc_channel_t)channel, duty);
  ledc_update_duty(SERVO_SPEED_MODE, (ledc_channel_t)channel);
}

void setWheelSpeeds(float left, float right) {
  // left/right expected range -1.0 .. 1.0
  if (fabs(left) < SERVO_DEADBAND) left = 0.0f;
  if (fabs(right) < SERVO_DEADBAND) right = 0.0f;
  left = constrain(left * LEFT_DIR, -1.0f, 1.0f);
  right = constrain(right * RIGHT_DIR, -1.0f, 1.0f);

  const int span = SERVO_MAX_US - SERVO_STOP_US;
  int left_us = SERVO_STOP_US + LEFT_TRIM_US + (int)(left * span);
  int right_us = SERVO_STOP_US + RIGHT_TRIM_US + (int)(right * span);

  writeServoUs(LEFT_SERVO_CHANNEL, left_us);
  writeServoUs(RIGHT_SERVO_CHANNEL, right_us);
}

void stopWheels() { setWheelSpeeds(0.0f, 0.0f); }

// =================
// HTML UI (served at /)
// =================
static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ESP32 Cam Rover</title>
  <style>
    :root { color-scheme: light; }
    body { margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: #0f172a; color: #e2e8f0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
    main { width: min(960px, 96vw); background: linear-gradient(135deg, #111827, #0b1a2c); border: 1px solid #1f2937; border-radius: 18px; padding: 18px 18px 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.35); }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    h1 { margin: 0; font-size: 1.25rem; letter-spacing: 0.02em; }
    #ip { font-size: 0.9rem; color: #a5b4fc; }
    #layout { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; }
    #stream-box { background: #0b1220; border: 1px solid #1f2937; border-radius: 14px; padding: 10px; display: flex; justify-content: center; align-items: center; }
    #stream { width: 100%; max-width: 480px; border-radius: 12px; background: #000; }
    #controls { display: grid; grid-template-rows: auto 1fr; gap: 10px; }
    .pad { display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 70px); gap: 8px; }
    button { border: 1px solid #334155; background: #1f2937; color: #e2e8f0; border-radius: 12px; font-size: 1rem; cursor: pointer; transition: transform 120ms ease, background 150ms ease, border-color 150ms ease; }
    button:active { transform: scale(0.98); background: #0ea5e9; border-color: #7dd3fc; color: #0b1220; }
    button[disabled] { opacity: 0.5; cursor: not-allowed; }
    .up { grid-column: 2; grid-row: 1; }
    .left { grid-column: 1; grid-row: 2; }
    .stop { grid-column: 2; grid-row: 2; background: #ef4444; border-color: #fecdd3; }
    .right { grid-column: 3; grid-row: 2; }
    .down { grid-column: 2; grid-row: 3; }
    .tip { font-size: 0.9rem; color: #94a3b8; margin-top: 6px; line-height: 1.4; }
    @media (max-width: 720px) { #layout { grid-template-columns: 1fr; } .pad { grid-template-rows: repeat(3, 60px); } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>ESP32-S3 Cam Rover</h1>
      <div id="ip">connecting...</div>
    </header>
    <section id="layout">
      <div id="stream-box">
        <img id="stream" alt="Camera stream" src="" />
      </div>
      <div id="controls">
        <div class="pad">
          <button class="up" data-cmd="fwd">▲</button>
          <button class="left" data-cmd="left">◀</button>
          <button class="stop" data-cmd="stop">■</button>
          <button class="right" data-cmd="right">▶</button>
          <button class="down" data-cmd="back">▼</button>
        </div>
        <div class="tip">
          Hold buttons or use keyboard: ↑ ↓ ← →. Release = stop.<br>
          Stream: 240×240 MJPEG (port 81). Drive API: /drive?l=[-1..1]&r=[-1..1]
        </div>
      </div>
    </section>
  </main>
  <script>
    const ipEl = document.getElementById('ip');
    const streamEl = document.getElementById('stream');
    const cmds = {
      fwd: { l: 1,   r: 1   },
      back:{ l:-1,   r:-1   },
      left:{ l:-0.55,r: 0.55},
      right:{l: 0.55,r:-0.55},
      stop:{ l: 0,   r: 0   }
    };

    const api = (l, r) => fetch(`/drive?l=${l}&r=${r}`).catch(() => {});

    function setStream() {
      const host = window.location.hostname;
      const url = `http://${host}:81/stream`;
      streamEl.src = url;
      ipEl.textContent = `${host} :80 / :81`;
    }

    function bindPad() {
      document.querySelectorAll('[data-cmd]').forEach(btn => {
        const cmd = btn.dataset.cmd;
        btn.addEventListener('pointerdown', e => { e.preventDefault(); const {l,r}=cmds[cmd]; api(l,r); });
        btn.addEventListener('pointerup', e => { e.preventDefault(); const {l,r}=cmds.stop; api(l,r); });
        btn.addEventListener('pointerleave', e => { if (e.pressure===0) { const {l,r}=cmds.stop; api(l,r); } });
      });
    }

    const keyMap = { ArrowUp:'fwd', ArrowDown:'back', ArrowLeft:'left', ArrowRight:'right' };
    const held = new Set();
    function handleKey(e) {
      const cmd = keyMap[e.key];
      if (!cmd) return;
      if (e.type === 'keydown' && e.repeat) return;
      e.preventDefault();
      if (e.type === 'keydown') {
        held.add(cmd);
        const {l,r}=cmds[cmd]; api(l,r);
      } else {
        held.delete(cmd);
        if (held.size === 0) { const {l,r}=cmds.stop; api(l,r); }
      }
    }
    window.addEventListener('keydown', handleKey);
    window.addEventListener('keyup', handleKey);

    setStream();
    bindPad();
  </script>
</body>
</html>
)rawliteral";

// ======================
// HTTP request handlers
// ======================
static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t drive_handler(httpd_req_t *req) {
  char query[64];
  float l = 0.0f, r = 0.0f;
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    char param[16];
    if (httpd_query_key_value(query, "l", param, sizeof(param)) == ESP_OK) {
      l = strtof(param, nullptr);
    }
    if (httpd_query_key_value(query, "r", param, sizeof(param)) == ESP_OK) {
      r = strtof(param, nullptr);
    }
    l = constrain(l, -1.0f, 1.0f);
    r = constrain(r, -1.0f, 1.0f);
  }
  setWheelSpeeds(l, r);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_sendstr(req, "OK");
}

static esp_err_t status_handler(httpd_req_t *req) {
  char json[96];
  snprintf(json, sizeof(json), "{\"left_dir\":%d,\"right_dir\":%d,\"left_trim\":%d,\"right_trim\":%d}", LEFT_DIR, RIGHT_DIR, LEFT_TRIM_US, RIGHT_TRIM_US);
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_sendstr(req, json);
}

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  struct timeval _timestamp;
  esp_err_t res = ESP_OK;
  size_t jpg_buf_len = 0;
  uint8_t *jpg_buf = NULL;
  char part_buf[128];

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      // If capture fails, try replaying the last good frame
      if (last_jpg_valid && last_jpg_buf && last_jpg_size > 0) {
        jpg_buf = last_jpg_buf;
        jpg_buf_len = last_jpg_size;
        _timestamp.tv_sec = 0;
        _timestamp.tv_usec = 0;
      } else {
        res = ESP_FAIL;
        break;
      }
    }

    if (fb) {
      if (fb->format != PIXFORMAT_JPEG) {
        bool jpeg_converted = frame2jpg(fb, 80, &jpg_buf, &jpg_buf_len);
        _timestamp.tv_sec = fb->timestamp.tv_sec;
        _timestamp.tv_usec = fb->timestamp.tv_usec;
        esp_camera_fb_return(fb);
        fb = NULL;
        if (!jpeg_converted) {
          res = ESP_FAIL;
        }
      } else {
        jpg_buf_len = fb->len;
        jpg_buf = fb->buf;
        _timestamp.tv_sec = fb->timestamp.tv_sec;
        _timestamp.tv_usec = fb->timestamp.tv_usec;
      }
    }

    if (res == ESP_OK && jpg_buf && jpg_buf_len > 0) {
      // copy to last buffer for fallback replay
      if (jpg_buf_len > last_jpg_cap) {
        uint8_t *newbuf = (uint8_t *)realloc(last_jpg_buf, jpg_buf_len);
        if (newbuf) {
          last_jpg_buf = newbuf;
          last_jpg_cap = jpg_buf_len;
        }
      }
      if (last_jpg_buf && jpg_buf_len <= last_jpg_cap) {
        memcpy(last_jpg_buf, jpg_buf, jpg_buf_len);
        last_jpg_size = jpg_buf_len;
        last_jpg_valid = true;
      }
    }

    if (res == ESP_OK) res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) {
      _timestamp.tv_sec = fb->timestamp.tv_sec;
      _timestamp.tv_usec = fb->timestamp.tv_usec;
      size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, jpg_buf_len, _timestamp.tv_sec, _timestamp.tv_usec);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_buf_len);

    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      jpg_buf = NULL;
    } else if (jpg_buf) {
      // if jpg_buf points to last_jpg_buf we don't free; only free conversions
      if (jpg_buf != last_jpg_buf && fb == NULL) {
        free(jpg_buf);
      }
      jpg_buf = NULL;
    }

    if (res != ESP_OK) {
      break;
    }
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.max_uri_handlers = 8;

  httpd_uri_t index_uri = {.uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = NULL};
  httpd_uri_t drive_uri = {.uri = "/drive", .method = HTTP_GET, .handler = drive_handler, .user_ctx = NULL};
  httpd_uri_t status_uri = {.uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL};

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &drive_uri);
    httpd_register_uri_handler(camera_httpd, &status_uri);
  }

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  stream_config.ctrl_port = 0;
  stream_config.max_uri_handlers = 2;
  stream_config.stack_size = 8192;
  stream_config.task_priority = tskIDLE_PRIORITY + 1;

  httpd_uri_t stream_uri = {.uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL};

  if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }

  // free any cached frame when server stops (defensive)
  if (last_jpg_buf) {
    free(last_jpg_buf);
    last_jpg_buf = NULL;
    last_jpg_cap = 0;
    last_jpg_size = 0;
    last_jpg_valid = false;
  }
}

// ============
// Setup & loop
// ============
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 16000000;
  config.frame_size = FRAMESIZE_240X240;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY; // smoother pacing; avoid tight polling
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 15; // higher number = more compression, less Wi-Fi load
  config.fb_count = 2;      // base buffer count

  if (psramFound()) {
    config.fb_count = 3;    // smoother stream bursts when PSRAM is available
  } else {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
    config.jpeg_quality = 17; // extra compression when DRAM-only
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_240X240);
  s->set_quality(s, 15); // match compression target above
  s->set_brightness(s, 0);
  s->set_contrast(s, 1);
  s->set_saturation(s, 1);
  s->set_sharpness(s, 1);
  s->set_lenc(s, 1);
  s->set_dcw(s, 1);
  // lock exposure/gain to reduce jitter; tweak if scene is too dark/bright


  // Servo PWM setup (LEDC, low-speed, timer 1)
  ledc_timer_config_t servo_timer = {
      .speed_mode = SERVO_SPEED_MODE,
      .duty_resolution = SERVO_RES_BITS,
      .timer_num = SERVO_TIMER,
      .freq_hz = (uint32_t)SERVO_FREQUENCY,
      .clk_cfg = LEDC_AUTO_CLK};
  ledc_timer_config(&servo_timer);

  ledc_channel_config_t left_ch = {
      .gpio_num = LEFT_SERVO_PIN,
      .speed_mode = SERVO_SPEED_MODE,
      .channel = LEFT_SERVO_CHANNEL,
      .intr_type = LEDC_INTR_DISABLE,
      .timer_sel = SERVO_TIMER,
      .duty = usToDuty(SERVO_STOP_US),
      .hpoint = 0};
  ledc_channel_config(&left_ch);

  ledc_channel_config_t right_ch = {
      .gpio_num = RIGHT_SERVO_PIN,
      .speed_mode = SERVO_SPEED_MODE,
      .channel = RIGHT_SERVO_CHANNEL,
      .intr_type = LEDC_INTR_DISABLE,
      .timer_sel = SERVO_TIMER,
      .duty = usToDuty(SERVO_STOP_US),
      .hpoint = 0};
  ledc_channel_config(&right_ch);
  stopWheels();

  WiFi.setSleep(false); // keep latency low
  WiFi.begin(ssid, password);
  Serial.printf("Connecting to %s", ssid);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  startCameraServer();
  Serial.println("HTTP server started");
  Serial.println("Open http://<IP> for controls, stream on http://<IP>:81/stream");
}

void loop() {
  delay(100);
}