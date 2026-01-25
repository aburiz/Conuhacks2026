import cv2
import numpy as np
import threading
import requests
import time
import os
import csv
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

# --- CONFIGURATION ---
ESP32_IP = "192.168.0.129"  # Update if your IP changes
STREAM_URL = f"http://{ESP32_IP}:81/stream"
BASE_URL = f"http://{ESP32_IP}"
SAVE_DIR = "dataset"

# Feature toggles
ENABLE_HAND_TRACKING = False
ENABLE_SOUND = False

# --- STATES ---
class AppMode(Enum):
    TELEOP = 1
    TRACKING = 2
    SENTRY = 3

# --- GLOBAL VARIABLES ---
current_mode = AppMode.TELEOP
current_action = [0.0, 0.0]  # [left, right]
recording = False
episode_path = None
csv_writer = None
csv_file = None
frame_count = 0

# Shared video frame
latest_frame = None
frame_lock = threading.Lock()
capture_running = threading.Event()

# --- NEW CONTROL WORKER (The Fix) ---
class ControlWorker:
    def __init__(self):
        self._running = True
        self._session = requests.Session()
        self._session.headers.update({"Connection": "keep-alive"})
        self._last_sent_action = [None, None]
        self._lock = threading.Lock()
        
        # Start the single worker thread
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            # 1. Get the LATEST global action
            # We copy it so we don't lock the main loop
            target = list(current_action)

            # 2. Optimization: Don't spam if nothing changed
            # (Unless we want a heartbeat, but keep-alive 0.1s is aggressive for HTTP)
            # We send if it changed OR if it's been >200ms (heartbeat)
            # For now, let's just limit rate to 20Hz (0.05s) to be safe
            
            try:
                # Send the request synchronously here. 
                # If network lags, THIS line waits. 
                # Meanwhile, 'current_action' in main loop keeps updating to the NEWEST value.
                url = f"{BASE_URL}/drive?l={target[0]:.2f}&r={target[1]:.2f}"
                self._session.get(url, timeout=0.2)
            except Exception as e:
                print(f"Cmd Error: {e}")
                time.sleep(0.1) # Back off on error

            # Limit control rate to ~20Hz to save ESP32 CPU for video
            time.sleep(0.05) 

    def stop(self):
        self._running = False

# Initialize the worker
control_worker = ControlWorker()

# --- (Keep your existing helper classes: HandTracker, SentryBrain, SoundNotifier) ---
# ... (Paste your classes here or keep them as is) ...
class HandTracker:
    def __init__(self, enabled): self.enabled = enabled
    def infer(self, frame): return None

class SentryBrain:
    def decide(self, frame): return (0.0, 0.0)

class SoundNotifier:
    def __init__(self, enabled): self.enabled = enabled
    def ping(self, event): pass

# --- (Keep your existing helper functions: start_recording, stop_recording, save_frame, draw_hud) ---
def start_recording():
    global episode_path, csv_writer, csv_file, recording, frame_count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_path = os.path.join(SAVE_DIR, timestamp)
    os.makedirs(episode_path, exist_ok=True)
    csv_file = open(os.path.join(episode_path, "data.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["filename", "v_left", "v_right", "timestamp"])
    recording = True
    frame_count = 0
    print(f"--- STARTED RECORDING: {episode_path} ---")

def stop_recording():
    global recording, csv_file, csv_writer
    if recording:
        recording = False
        if csv_file:
            csv_file.close()
            csv_file = None
            csv_writer = None
        print(f"--- STOPPED RECORDING. Saved {frame_count} frames. ---")

def save_frame(frame, action):
    global frame_count
    if not recording or not episode_path: return
    timestamp_ms = int(time.time() * 1000)
    filename = f"{timestamp_ms}.jpg"
    cv2.imwrite(os.path.join(episode_path, filename), frame)
    if csv_writer:
        csv_writer.writerow([filename, action[0], action[1], timestamp_ms])
        frame_count += 1

def draw_hud(frame):
    h, w, _ = frame.shape
    cv2.putText(frame, f"MODE: {current_mode.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if recording:
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Motor Bars
    l_val, r_val = int(current_action[0] * 50), int(current_action[1] * 50)
    cv2.rectangle(frame, (10, h - 20), (110, h - 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (60, h - 20), (60 + l_val, h - 10), (0, 255, 0), -1)
    cv2.rectangle(frame, (120, h - 20), (220, h - 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (170, h - 20), (170 + r_val, h - 10), (0, 255, 0), -1)

# --- KEYBOARD INPUT (Simplified) ---
def decode_drive_key(key):
    # Mapping for arrow keys and WASD
    # (Simplified for clarity, your existing logic is fine too)
    if key == -1: return None
    # WASD
    if key == ord('w'): return (1.0, 1.0)
    if key == ord('s'): return (-1.0, -1.0)
    if key == ord('a'): return (-0.5, 0.5)
    if key == ord('d'): return (0.5, -0.5)
    # Arrows (Linux/Win common codes)
    if key in [82, 0x260000]: return (1.0, 1.0) # Up
    if key in [84, 0x280000]: return (-1.0, -1.0) # Down
    if key in [81, 0x250000]: return (-0.5, 0.5) # Left
    if key in [83, 0x270000]: return (0.5, -0.5) # Right
    return None

def main():
    global current_action, current_mode, recording, latest_frame

    # Start Capture Thread
    cap = cv2.VideoCapture(STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Critical for low latency video
    
    if not cap.isOpened():
        print("Error connecting to camera")
        return
    capture_running.set()

    def capture_loop():
        global latest_frame
        while capture_running.is_set():
            ok, frame = cap.read()
            if ok:
                with frame_lock:
                    latest_frame = frame
            else:
                # If stream drops, don't hammer the CPU
                time.sleep(0.01)

    threading.Thread(target=capture_loop, daemon=True).start()

    print("\n--- CONTROLS: WASD / Arrows. SPACE to Record. Q to Quit. ---")

    while True:
        # 1. Frame Handling
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(frame, "CONNECTING...", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # 2. Data Collection (Save Frame + Label)
        if current_mode == AppMode.TELEOP and recording:
            save_img = cv2.resize(frame, (224, 224))
            save_frame(save_img, current_action)

        # 3. Visualization
        draw_hud(frame)
        cv2.imshow('ESP32 Controller', frame)

        # 4. Input Handling
        key = cv2.waitKeyEx(1)
        if key == ord('q'):
            break
        elif key == 32: # SPACE
            if recording: stop_recording()
            else: start_recording()
        
        # Drive Logic
        # Note: We just update the GLOBAL 'current_action'. 
        # The background ControlWorker picks this up automatically.
        cmd = decode_drive_key(key)
        if cmd:
            current_action = list(cmd)
        else:
            # If no key pressed, stop (Zero-order hold)
            current_action = [0.0, 0.0]

    # Cleanup
    control_worker.stop()
    capture_running.clear()
    cap.release()
    cv2.destroyAllWindows()
    if csv_file: csv_file.close()

if __name__ == '__main__':
    main()