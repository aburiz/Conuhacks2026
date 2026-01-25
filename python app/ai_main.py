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

# --- NEW IMPORTS FOR INFERENCE ---
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# --- CONFIGURATION ---
ESP32_IP = "192.168.0.129"  # Update if your IP changes
STREAM_URL = f"http://{ESP32_IP}:81/stream"
BASE_URL = f"http://{ESP32_IP}"
SAVE_DIR = "dataset"
USE_UDP = True
UDP_PORT = 3333
MODEL_PATH = "output/sentry_policy_20260124_214033.pth"   # <--- Path to your trained model

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

# Networking
control_session = requests.Session()
control_session.headers.update({"Connection": "keep-alive"})

# Timing
CMD_KEEPALIVE = 0.10   

# Shared Data
latest_frame = None
frame_lock = threading.Lock()
capture_running = threading.Event()
sender_running = threading.Event()
desired_action = [0.0, 0.0]
udp_sock = None

# Async Keys (Windows)
try:
    import ctypes
    _GetKeyState = ctypes.windll.user32.GetAsyncKeyState
    _HAS_ASYNC = True
except Exception:
    _GetKeyState = None
    _HAS_ASYNC = False

# --- MODEL DEFINITION (Must match training exactly) ---
class SentryPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # We need to match the chunk_size used in training to define the output layer correctly
        self.chunk_size = 10 
        
        # Initialize backbone exactly as in training
        # Note: weights=None is fine here since we are loading a state_dict immediately after
        self.backbone = resnet18(weights=None) 
        self.backbone.fc = nn.Identity() 
        
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.chunk_size * 2) 
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# --- CLASSES ---

class HandTracker:
    """Placeholder for MediaPipe hand tracking."""
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def infer(self, frame) -> Optional[Tuple[float, float]]:
        if not self.enabled:
            return None
        return None

class SentryBrain:
    """Runs the trained ResNet18 policy."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        print(f"--- SENTRY BRAIN INITIALIZING ON {self.device.upper()} ---")
        
        if os.path.exists(MODEL_PATH):
            try:
                # 1. Instantiate the architecture
                self.model = SentryPolicy()
                
                # 2. Load the trained weights
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # 3. Prep for inference
                self.model.to(self.device)
                self.model.eval()
                print(f"SUCCESS: Loaded model from {MODEL_PATH}")
            except Exception as e:
                print(f"ERROR: Failed to load model: {e}")
                self.model = None
        else:
            print(f"WARNING: No model found at {MODEL_PATH}. Sentry mode will do nothing.")

    def decide(self, frame) -> Tuple[float, float]:
        if self.model is None or frame is None:
            return (0.0, 0.0)

        # 1. Preprocess exactly like Training (Resize -> Tensor -> Normalize)
        # Resize to 240x240
        img = cv2.resize(frame, (240, 240))
        
        # Convert to Tensor, Permute to (C, H, W), Normalize
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Add Batch Dimension (1, 3, 240, 240)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 2. Inference
        with torch.no_grad():
            output = self.model(img_tensor) # Output shape: (1, 20)
            
        # 3. Post-process
        # The model outputs a chunk of 10 actions: [v_l1, v_r1, v_l2, v_r2, ...]
        # We take the FIRST action pair for immediate execution (Receding Horizon Control)
        actions = output.cpu().numpy().flatten()
        
        v_left = float(actions[0])
        v_right = float(actions[1])
        
        # Optional: Clamp values to safety limits [-1, 1]
        v_left = max(-1.0, min(1.0, v_left))
        v_right = max(-1.0, min(1.0, v_right))

        return (v_left, v_right)

class SoundNotifier:
    """Placeholder for sound events."""
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def ping(self, event: str):
        if not self.enabled: return
        pass

# --- HELPER FUNCTIONS ---

def send_command(left, right):
    """HTTP fallback for drive commands."""
    try:
        url = f"{BASE_URL}/drive?l={left:.2f}&r={right:.2f}"
        control_session.get(url, timeout=0.08)
    except requests.exceptions.RequestException:
        pass 

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
    path = os.path.join(episode_path, filename)
    cv2.imwrite(path, frame)
    if csv_writer:
        csv_writer.writerow([filename, action[0], action[1], timestamp_ms])
        frame_count += 1

def draw_hud(frame):
    h, w, _ = frame.shape
    mode_text = f"MODE: {current_mode.name}"
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if recording:
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (w - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Action Bars
    l_val = int(current_action[0] * 50)
    cv2.rectangle(frame, (10, h - 20), (10 + 100, h - 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (60, h - 20), (60 + l_val, h - 10), (0, 255, 0), -1)
    
    r_val = int(current_action[1] * 50)
    cv2.rectangle(frame, (120, h - 20), (220, h - 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (170, h - 20), (170 + r_val, h - 10), (0, 255, 0), -1)

    cv2.putText(frame, f"L: {current_action[0]:.2f}", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"R: {current_action[1]:.2f}", (120, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def decode_drive_key(key: int) -> Optional[Tuple[float, float]]:
    key_low = key & 0xFF
    if key_low == 82 or key in (0x260000, 2490368) or key_low == ord('w'): return (1.0, 1.0)
    if key_low == 84 or key in (0x280000, 2621440) or key_low == ord('s'): return (-1.0, -1.0)
    if key_low == 81 or key in (0x250000, 2424832) or key_low == ord('a'): return (-0.5, 0.5)
    if key_low == 83 or key in (0x270000, 2555904) or key_low == ord('d'): return (0.5, -0.5)
    return None

def poll_drive_state() -> Optional[Tuple[float, float]]:
    if not _HAS_ASYNC: return None
    pressed = lambda vk: (_GetKeyState(vk) & 0x8000) != 0
    if pressed(0x26) or pressed(ord('W')): return (1.0, 1.0)
    if pressed(0x28) or pressed(ord('S')): return (-1.0, -1.0)
    if pressed(0x25) or pressed(ord('A')): return (-0.5, 0.5)
    if pressed(0x27) or pressed(ord('D')): return (0.5, -0.5)
    return None

# --- MAIN LOOP ---

def main():
    global current_action, current_mode, recording, udp_sock

    last_sent_action = [None, None]
    last_sent_action_time = 0.0
    notifier = SoundNotifier(ENABLE_SOUND)
    hand_tracker = HandTracker(ENABLE_HAND_TRACKING)
    
    # Initialize Sentry Brain (Loads Model)
    sentry = SentryBrain()

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

    # Thread 1: Video Capture
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

    # Thread 2: Command Sender (UDP/HTTP)
    def sender_loop():
        global last_sent_action_time
        while sender_running.is_set():
            now = time.time()
            da = list(desired_action)
            if da != last_sent_action or now - last_sent_action_time > CMD_KEEPALIVE:
                try:
                    if USE_UDP and udp_sock:
                        payload = f"{da[0]:.2f},{da[1]:.2f}".encode()
                        udp_sock.sendto(payload, (ESP32_IP, UDP_PORT))
                    else:
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

        # --- MODE HANDLING ---
        
        # 1. Teleoperation
        if current_mode == AppMode.TELEOP:
            if recording:
                save_img = cv2.resize(frame, (224, 224))
                save_frame(save_img, current_action)

            gesture_action = hand_tracker.infer(frame)
            if gesture_action is not None:
                current_action = list(gesture_action)
        
        # 2. Tracking (Placeholder)
        elif current_mode == AppMode.TRACKING:
            current_action = [0.0, 0.0]
            cv2.putText(frame, "TRACKING (In Dev)", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3. Sentry Mode (AI Pilot)
        elif current_mode == AppMode.SENTRY:
            # Inference happens here
            auto_action = sentry.decide(frame)
            current_action = list(auto_action)
            
            # Visual Feedback
            cv2.putText(frame, "SENTRY AI ACTIVE", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Inf: {current_action[0]:.2f}, {current_action[1]:.2f}", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        draw_hud(frame)
        cv2.imshow('ESP32 RC Controller', frame)

        # --- INPUT HANDLING ---
        drive_state = poll_drive_state()
        key = cv2.waitKeyEx(1)

        if key == ord('q'):
            stop_recording()
            break
        elif key in (9, ord('1'), ord('2'), ord('3')):
            if key == ord('1'): current_mode = AppMode.TELEOP
            elif key == ord('2'): current_mode = AppMode.TRACKING
            elif key == ord('3'): current_mode = AppMode.SENTRY
            else: current_mode = AppMode((current_mode.value % 3) + 1)
            
            stop_recording()
            current_action = [0.0, 0.0]
            notifier.ping("mode")
            print(f"Switched to {current_mode.name}")

        # Drive Inputs only override in Teleop
        if current_mode == AppMode.TELEOP:
            if key == 32:  # SPACE
                if recording: stop_recording()
                else: start_recording()
            
            drive = drive_state or decode_drive_key(key)
            if drive:
                current_action = list(drive)
            elif drive_state is None and key == -1 and gesture_action is None:
                current_action = [0.0, 0.0]

        # Update global desired action for the sender thread
        desired_action[0], desired_action[1] = current_action[0], current_action[1]

    capture_running.clear()
    sender_running.clear()
    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()

if __name__ == '__main__':
    main()