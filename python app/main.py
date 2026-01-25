import cv2
import numpy as np
import urllib.request
import threading
import requests
import time
import os
import csv
import socket
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

# --- CONFIGURATION ---
ESP32_IP = "192.168.0.129"  # Update if your IP changes
STREAM_URL = f"http://{ESP32_IP}:81/stream"
BASE_URL = f"http://{ESP32_IP}"
SAVE_DIR = "dataset"
USE_UDP = True
UDP_PORT = 3333

# Feature toggles / placeholders
ENABLE_HAND_TRACKING = False  # hook MediaPipe here later
ENABLE_SOUND = False          # hook sound feedback here later

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

# Networking session for faster repeated requests
control_session = requests.Session()
control_session.headers.update({"Connection": "keep-alive"})

# Timing for control responsiveness
CMD_KEEPALIVE = 0.10   # seconds between resends of same command when held

# Shared video frame
latest_frame = None
frame_lock = threading.Lock()
capture_running = threading.Event()
sender_running = threading.Event()
desired_action = [0.0, 0.0]
udp_sock = None

# Optional async key polling (Windows)
try:
    import ctypes

    _GetKeyState = ctypes.windll.user32.GetAsyncKeyState
    _HAS_ASYNC = True
except Exception:
    _GetKeyState = None
    _HAS_ASYNC = False
# Placeholders ---------------------------------------------------------------
class HandTracker:
    """Placeholder for MediaPipe hand tracking."""
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def infer(self, frame) -> Optional[Tuple[float, float]]:
        """Return (left,right) in [-1,1] or None if no hand intent."""
        if not self.enabled:
            return None
        # TODO: implement MediaPipe Hands -> drive mapping
        return None


class SentryBrain:
    """Placeholder for autonomous sentry/maze logic."""
    def decide(self, frame) -> Tuple[float, float]:
        if frame is None:
            return (0.0, 0.0)
        # TODO: replace with RL/heuristic policy
        return (0.0, 0.0)


class SoundNotifier:
    """Placeholder for sound events."""
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def ping(self, event: str):
        if not self.enabled:
            return
        # TODO: play tone based on event
        pass

def send_command(left, right):
    """HTTP fallback for drive commands."""
    try:
        url = f"{BASE_URL}/drive?l={left:.2f}&r={right:.2f}"
        control_session.get(url, timeout=0.08)
    except requests.exceptions.RequestException:
        pass  # Ignore drops to keep video smooth

def start_recording():
    """Initializes a new episode folder and CSV file."""
    global episode_path, csv_writer, csv_file, recording, frame_count
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_path = os.path.join(SAVE_DIR, timestamp)
    os.makedirs(episode_path, exist_ok=True)
    
    # Create CSV
    csv_file = open(os.path.join(episode_path, "data.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["filename", "v_left", "v_right", "timestamp"])
    
    recording = True
    frame_count = 0
    print(f"--- STARTED RECORDING: {episode_path} ---")

def stop_recording():
    """Closes the CSV and stops recording."""
    global recording, csv_file, csv_writer
    if recording:
        recording = False
        if csv_file:
            csv_file.close()
            csv_file = None
            csv_writer = None
        print(f"--- STOPPED RECORDING. Saved {frame_count} frames. ---")

def save_frame(frame, action):
    """Saves the current frame and action to disk."""
    global frame_count
    if not recording or not episode_path:
        return

    timestamp_ms = int(time.time() * 1000)
    filename = f"{timestamp_ms}.jpg"
    
    # Save Image
    path = os.path.join(episode_path, filename)
    cv2.imwrite(path, frame)
    
    # Save Data
    if csv_writer:
        csv_writer.writerow([filename, action[0], action[1], timestamp_ms])
        framsqe_count += 1

def draw_hud(frame):
    """Draws the overlay UI on the frame."""
    h, w, _ = frame.shape
    
    # 1. Mode Indicator
    mode_text = f"MODE: {current_mode.name}"
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 2. Recording Indicator
    if recording:
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (w - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 3. Action Bars (Visualizing motor output)
    # Left Motor
    l_val = int(current_action[0] * 50)
    cv2.rectangle(frame, (10, h - 20), (10 + 100, h - 10), (50, 50, 50), -1) # BG
    cv2.rectangle(frame, (60, h - 20), (60 + l_val, h - 10), (0, 255, 0), -1) # Bar
    
    # Right Motor
    r_val = int(current_action[1] * 50)
    cv2.rectangle(frame, (120, h - 20), (220, h - 10), (50, 50, 50), -1) # BG
    cv2.rectangle(frame, (170, h - 20), (170 + r_val, h - 10), (0, 255, 0), -1) # Bar

    cv2.putText(frame, f"L: {current_action[0]}", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"R: {current_action[1]}", (120, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def decode_drive_key(key: int) -> Optional[Tuple[float, float]]:
    """Map keycode from cv2.waitKeyEx to differential drive command."""
    key_low = key & 0xFF
    # Arrow keys (Linux/mac typical)
    if key_low == 82:   # Up
        return (1.0, 1.0)
    if key_low == 84:   # Down
        return (-1.0, -1.0)
    if key_low == 81:   # Left
        return (-0.5, 0.5)
    if key_low == 83:   # Right
        return (0.5, -0.5)
    # Arrow keys (Windows virtual-key in waitKeyEx)
    if key in (0x260000, 2490368):  # Up
        return (1.0, 1.0)
    if key in (0x280000, 2621440):  # Down
        return (-1.0, -1.0)
    if key in (0x250000, 2424832):  # Left
        return (-0.5, 0.5)
    if key in (0x270000, 2555904):  # Right
        return (0.5, -0.5)
    # WASD fallback
    if key_low == ord('w'):
        return (1.0, 1.0)
    if key_low == ord('s'):
        return (-1.0, -1.0)
    if key_low == ord('a'):
        return (-0.5, 0.5)
    if key_low == ord('d'):
        return (0.5, -0.5)
    return None


def poll_drive_state() -> Optional[Tuple[float, float]]:
    """Instantaneous key state (Windows)."""
    if not _HAS_ASYNC:
        return None
    pressed = lambda vk: (_GetKeyState(vk) & 0x8000) != 0
    # Prioritize arrows
    if pressed(0x26):  # Up
        return (1.0, 1.0)
    if pressed(0x28):  # Down
        return (-1.0, -1.0)
    if pressed(0x25):  # Left
        return (-0.5, 0.5)
    if pressed(0x27):  # Right
        return (0.5, -0.5)
    if pressed(ord('W')):
        return (1.0, 1.0)
    if pressed(ord('S')):
        return (-1.0, -1.0)
    if pressed(ord('A')):
        return (-0.5, 0.5)
    if pressed(ord('D')):
        return (0.5, -0.5)
    return None
def main():
    global current_action, current_mode, recording

    last_sent_action = [None, None]
    last_sent_action_time = 0.0
    notifier = SoundNotifier(ENABLE_SOUND)
    hand_tracker = HandTracker(ENABLE_HAND_TRACKING)
    sentry = SentryBrain()

    # UDP setup
    global udp_sock
    if USE_UDP:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setblocking(False)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Connecting to stream at: {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Error connecting to camera")
        return
    capture_running.set()
    sender_running.set()

    def capture_loop():
        global latest_frame
        while capture_running.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                with frame_lock:
                    latest_frame = frame
            else:
                time.sleep(0.005)

    threading.Thread(target=capture_loop, daemon=True).start()
    def sender_loop():
        global last_sent_action_time
        while sender_running.is_set():
            now = time.time()
            # Snapshot desired action
            da = list(desired_action)
            if da != last_sent_action or now - last_sent_action_time > CMD_KEEPALIVE:
                try:
                    if USE_UDP and udp_sock:
                        payload = f"{da[0]:.2f},{da[1]:.2f}".encode()
                        udp_sock.sendto(payload, (ESP32_IP, UDP_PORT))
                    else:
                        send_command(da[0], da[1])
                except Exception:
                    # retry once after brief pause via HTTP fallback
                    time.sleep(0.03)
                    try:
                        send_command(da[0], da[1])
                    except Exception:
                        pass
                last_sent_action[:] = da
                last_sent_action_time = now
            time.sleep(0.01)

    threading.Thread(target=sender_loop, daemon=True).start()

    print("\n--- CONTROLS ---")
    print("TAB / 1/2/3 : Switch modes (Teleop / Tracking / Sentry)")
    print("SPACE       : Toggle recording")
    print("Arrows/WASD : Drive (Teleop)")
    print("Q           : Quit")

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else blank

        if current_mode == AppMode.TELEOP:
            if recording:
                save_img = cv2.resize(frame, (224, 224))
                save_frame(save_img, current_action)

            gesture_action = hand_tracker.infer(frame)
            if gesture_action is not None:
                current_action = list(gesture_action)

            if current_action != last_sent_action:
                threading.Thread(target=send_command, args=(current_action[0], current_action[1])).start()
                last_sent_action = list(current_action)

        elif current_mode == AppMode.TRACKING:
            if current_action != [0.0, 0.0]:
                current_action = [0.0, 0.0]
                send_command(0, 0)
                last_sent_action = [0.0, 0.0]
            cv2.putText(frame, "TRACKING PLACEHOLDER (hand tracking soon)", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        elif current_mode == AppMode.SENTRY:
            auto_action = sentry.decide(frame)
            current_action = list(auto_action)
            if current_action != last_sent_action:
                threading.Thread(target=send_command, args=(current_action[0], current_action[1])).start()
                last_sent_action = list(current_action)
            cv2.putText(frame, "SENTRY PLACEHOLDER", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        draw_hud(frame)
        cv2.imshow('ESP32 RC Controller', frame)

        drive_state = poll_drive_state()
        key = cv2.waitKeyEx(1)

        if key == ord('q'):
            stop_recording()
            break
        elif key in (9, ord('1'), ord('2'), ord('3')):  # TAB or numeric
            if key == ord('1'):
                current_mode = AppMode.TELEOP
            elif key == ord('2'):
                current_mode = AppMode.TRACKING
            elif key == ord('3'):
                current_mode = AppMode.SENTRY
            else:
                current_mode = AppMode((current_mode.value % 3) + 1)
            stop_recording()
            current_action = [0.0, 0.0]
            send_command(0, 0)
            last_sent_action = [0.0, 0.0]
            last_sent_action_time = time.time()
            notifier.ping("mode")
            print(f"Switched to {current_mode.name}")

        if current_mode == AppMode.TELEOP:
            if key == 32:  # SPACE
                if recording:
                    stop_recording()
                else:
                    start_recording()
            drive = drive_state or decode_drive_key(key)
            if drive:
                current_action = list(drive)
            elif drive_state is None and key == -1:
                current_action = [0.0, 0.0]

        # update desired action for sender thread
        desired_action[0], desired_action[1] = current_action[0], current_action[1]

    capture_running.clear()
    sender_running.clear()
    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()


if __name__ == '__main__':
    main()
