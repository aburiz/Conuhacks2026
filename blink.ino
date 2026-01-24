// Minimal blink for Seeed Studio XIAO ESP32S3 Sense.
// On-board user LED is wired to GPIO21 with inverted logic (LOW = on).

#include <Arduino.h>

constexpr uint8_t LED_PIN = 21;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // start with LED off (active LOW)
}

void loop() {
  digitalWrite(LED_PIN, LOW);   // LED on
  delay(500);
  digitalWrite(LED_PIN, HIGH);  // LED off
  delay(500);
}
