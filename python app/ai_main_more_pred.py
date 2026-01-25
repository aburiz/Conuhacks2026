import cv2
import numpy as np
import urllib.request
import threading
import requests
import time
import os
import csv
import socket
import collections # <--- Added for Action Buffer
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, List # <--- Added List
import math
from pathlib import Path

# --- AI IMPORTS ---
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- CONFIGURATION ---
ESP32_IP = "192.168.0.129"  # Update if your IP changes
STREAM_URL = f"http://{ESP32_IP}:81/stream"
BASE_URL = f"http://{ESP32_IP}"
SAVE_DIR = "dataset"
USE_UDP = True
UDP_PORT = 3333

# Models
# Update this filename to your latest .pth
SENTRY_MODEL_PATH = "output/sentry_policy_20260125_034803.pth" 
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

DRIVE_SCALE = 0.2  # scale outgoing drive commands without reflashing ESP
MAX_FPS = 30
FRAME_PERIOD = 1.0 / MAX_FPS

# NOTE: This must match your training! 
# The file provided earlier used 10, but if you retrained for 30, change this to 30.
CHUNK_SIZE = 10 

# Feature toggles
ENABLE_HAND_TRACKING = True   # MediaPipe hand tracking for gesture drive
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

# --- ACTION BUFFER (The Lag Killer) ---
action_queue = collections.deque(maxlen=CHUNK_SIZE) 
queue_lock = threading.Lock()

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

# ---------------------------------------------------------------------------
# MediaPipe model utilities (Unchanged)
# ---------------------------------------------------------------------------
def ensure_model():
    if MODEL_PATH.exists():
        return
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

try:
    import ctypes
    _GetKeyState = ctypes.windll.user32.GetAsyncKeyState
    _HAS_ASYNC = True
except Exception:
    _GetKeyState = None
    _HAS_ASYNC = False

# ---------------------------------------------------------------------------
# AI Model Definitions (MATCHING TRAIN SCRIPT)
# ---------------------------------------------------------------------------
class SentryPolicy(nn.Module):
    """
    The ResNet Policy Architecture.
    STRICTLY MATCHES 'train_ResNet_18_architecture.py'
    """
    def __init__(self):
        super().__init__()
        self.chunk_size = CHUNK_SIZE 
        # Note: weights=None is fine here since we load state_dict immediately
        self.backbone = resnet18(weights=None) 
        self.backbone.fc = nn.Identity() 
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.chunk_size * 2)
            # NO TANH HERE (Matches training script)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class SentryBrain:
    """
    Handles loading the model and running inference on frames.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        print(f"--- SENTRY BRAIN INITIALIZING ON {self.device.upper()} ---")
        
        if os.path.exists(SENTRY_MODEL_PATH):
            try:
                self.model = SentryPolicy()
                state_dict = torch.load(SENTRY_MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print(f"SUCCESS: Loaded Sentry AI from {SENTRY_MODEL_PATH}")
            except Exception as e:
                print(f"ERROR: Failed to load Sentry AI: {e}")
                self.model = None
        else:
            print(f"WARNING: No model found at {SENTRY_MODEL_PATH}. Sentry mode will do nothing.")

    def decide(self, frame) -> List[Tuple[float, float]]:
        if self.model is None or frame is None:
            return []

        # 1. Resize to match training input (240x240)
        img = cv2.resize(frame, (240, 240))
        
        # 2. COLOR CORRECTION (BGR -> RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Normalize & Tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # 4. Normalize with ImageNet stats
        img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / \
                     torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 5. Inference
        with torch.no_grad():
            output = self.model(img_tensor) 
            
        # 6. Parse ALL predictions
        flat = output.cpu().numpy().flatten()
        
        actions = []
        for i in range(0, len(flat), 2):
            v_left = float(flat[i])
            v_right = float(flat[i+1])
            
            # Clip safely
            v_left = max(-1.0, min(1.0, v_left))
            v_right = max(-1.0, min(1.0, v_right))
            
            actions.append((v_left, v_right))
            
        return actions

# ---------------------------------------------------------------------------
# Hand Tracking Logic (Unchanged)
# ---------------------------------------------------------------------------
class HandTracker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.present = False
        self.missing_frames = 0
        self.max_missing = 5
        try:
            ensure_model()
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
            self.enabled = enabled
        except Exception as e:
            print(f"[HandTracker] MediaPipe not available ({e}); hand control disabled.")
            self.enabled = False
            self.landmarker = None

        self.open_thresh = 0.18
        self.fist_thresh = 0.12
        self.smooth = 0.3
        self.prev = (0.0, 0.0)
        self.t0 = time.perf_counter()
        self.deadband = 0.05
        self.quant_step = 0.05

    def infer(self, frame) -> Optional[Tuple[float, float]]:
        if not self.enabled or self.landmarker is None:
            return None
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - self.t0) * 1000)
        res = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not res.hand_landmarks:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing:
                self.prev = (0.0, 0.0)
            return None

        self.missing_frames = 0
        hand = res.hand_landmarks[0]
        wrist = hand[0]
        tips = [hand[i] for i in (8, 12, 16, 20)]
        
        mean_tip_dist = sum(math.hypot(t.x - wrist.x, t.y - wrist.y) for t in tips) / len(tips)
        is_fist = mean_tip_dist < self.fist_thresh
        is_open = mean_tip_dist > self.open_thresh

        if is_fist:
            self.prev = (0.0, 0.0)
            return (0.0, 0.0)

        if not is_open:
            return None

        forward = (0.5 - wrist.y) * 2.0  
        turn = (0.5 - wrist.x) * 2.0     
        forward = max(-1.0, min(1.0, forward))
        turn = max(-1.0, min(1.0, turn))
        left = forward - turn
        right = forward + turn
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        l = self.smooth * left + (1 - self.smooth) * self.prev[0]
        r = self.smooth * right + (1 - self.smooth) * self.prev[1]

        def q(v):
            if abs(v) < self.deadband: return 0.0
            return round(v / self.quant_step) * self.quant_step

        l = q(l)
        r = q(r)
        self.prev = (l, r)
        return (l, r)

class SoundNotifier:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def ping(self, event: str):
        pass

# ---------------------------------------------------------------------------
# UPDATED SENDER LOOP
# ---------------------------------------------------------------------------
def send_command(left, right):
    try:
        url = f"{BASE_URL}/drive?l={left:.2f}&r={right:.2f}"
        control_session.get(url, timeout=0.08)
    except requests.exceptions.RequestException:
        pass

def sender_loop():
    global current_action
    
    while sender_running.is_set():
        start_time = time.time()
        
        target_l, target_r = 0.0, 0.0
        
        if current_mode == AppMode.SENTRY:
            # Consume from Buffer
            with queue_lock:
                if len(action_queue) > 0:
                    target_l, target_r = action_queue.popleft()
                    current_action = [target_l, target_r] 
                else:
                    # Buffer empty -> Stop
                    target_l, target_r = 0.0, 0.0
        else:
            # Manual Mode
            target_l = current_action[0]
            target_r = current_action[1]

        final_l = target_l * DRIVE_SCALE
        final_r = target_r * DRIVE_SCALE
        
        try:
            if USE_UDP and udp_sock:
                payload = f"{final_l:.2f},{final_r:.2f}".encode()
                udp_sock.sendto(payload, (ESP32_IP, UDP_PORT))
            else:
                send_command(final_l, final_r)
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed < FRAME_PERIOD:
            time.sleep(FRAME_PERIOD - elapsed)

# ---------------------------------------------------------------------------
# UTILS (Recording / Input)
# ---------------------------------------------------------------------------
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
    if not recording or not episode_path:
        return
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
    if key_low == 82: return (1.0, 1.0)
    if key_low == 84: return (-1.0, -1.0)
    if key_low == 81: return (-0.5, 0.5)
    if key_low == 83: return (0.5, -0.5)
    if key in (0x260000, 2490368): return (1.0, 1.0)
    if key in (0x280000, 2621440): return (-1.0, -1.0)
    if key in (0x250000, 2424832): return (-0.5, 0.5)
    if key in (0x270000, 2555904): return (0.5, -0.5)
    if key_low == ord('w'): return (1.0, 1.0)
    if key_low == ord('s'): return (-1.0, -1.0)
    if key_low == ord('a'): return (-0.5, 0.5)
    if key_low == ord('d'): return (0.5, -0.5)
    return None

def poll_drive_state() -> Optional[Tuple[float, float]]:
    if not _HAS_ASYNC: return None
    pressed = lambda vk: (_GetKeyState(vk) & 0x8000) != 0
    if pressed(0x26): return (1.0, 1.0)
    if pressed(0x28): return (-1.0, -1.0)
    if pressed(0x25): return (-0.5, 0.5)
    if pressed(0x27): return (0.5, -0.5)
    if pressed(ord('W')): return (1.0, 1.0)
    if pressed(ord('S')): return (-1.0, -1.0)
    if pressed(ord('A')): return (-0.5, 0.5)
    if pressed(ord('D')): return (0.5, -0.5)
    return None

def main():
    global current_action, current_mode, recording, udp_sock

    last_sent_action = [None, None]
    last_sent_action_time = 0.0
    notifier = SoundNotifier(ENABLE_SOUND)
    hand_tracker = HandTracker(ENABLE_HAND_TRACKING)
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
    threading.Thread(target=sender_loop, daemon=True).start()

    print("\n--- CONTROLS ---")
    cv2.namedWindow('ESP32 RC Controller', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ESP32 RC Controller', 1920, 1080)

    print("TAB / 1/2/3 : Switch modes (Teleop / Tracking / Sentry)")
    print("SPACE       : Toggle recording")
    print("Arrows/WASD : Drive (Teleop)")
    print("Q           : Quit")

    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    last_frame_time = time.time()
    
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else blank

        now = time.time()
        dt = now - last_frame_time
        if dt < FRAME_PERIOD:
            time.sleep(FRAME_PERIOD - dt)
        last_frame_time = time.time()

        if current_mode == AppMode.SENTRY:
            # 1. Inference
            new_chunk = sentry.decide(frame)
            
            # 2. Update Buffer (Overwrite old plan)
            if new_chunk:
                with queue_lock:
                    action_queue.clear()
                    action_queue.extend(new_chunk)
                    
            cv2.putText(frame, f"SENTRY: Buffer {len(action_queue)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif current_mode == AppMode.TRACKING:
            gesture_action = hand_tracker.infer(frame)
            if gesture_action is not None:
                current_action = list(gesture_action)
            else:
                current_action = [0.0, 0.0]
            cv2.putText(frame, "TRACKING (gesture control)", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        elif current_mode == AppMode.TELEOP:
            if recording:
                save_img = cv2.resize(frame, (224, 224))
                save_frame(save_img, current_action)

        draw_hud(frame)
        target_h = 1080
        fh, fw, _ = frame.shape
        scale = target_h / float(fh)
        target_w = int(fw * scale)
        display = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('ESP32 RC Controller', display)

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
            send_command(0, 0)
            notifier.ping("mode")
            print(f"Switched to {current_mode.name}")

        if current_mode == AppMode.TELEOP:
            if key == 32: 
                if recording: stop_recording()
                else: start_recording()
            drive = drive_state or decode_drive_key(key)
            if drive:
                current_action = list(drive)
            elif drive_state is None and key == -1:
                current_action = [0.0, 0.0]

    capture_running.clear()
    sender_running.clear()
    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()

if __name__ == '__main__':
    main()