import cv2
import numpy as np
import urllib.request
import threading
import requests
import time
import os
import csv
import socket
import collections
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, List
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
SENTRY_MODEL_PATH = "output/advanced/best_memory_model.pth" # <--- POINT TO NEW MODEL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Drive Settings
DRIVE_SCALE = 1.0  # Set to 1.0 because we use clean_motor_command to limit speed now
MAX_FPS = 30
FRAME_PERIOD = 1.0 / MAX_FPS

# AI Settings (MUST MATCH TRAINING)
CHUNK_SIZE = 20
SEQ_LEN = 5  # Length of memory (How many past frames to look at)

# Feature toggles
ENABLE_HAND_TRACKING = True
ENABLE_SOUND = False

# --- STATES ---
class AppMode(Enum):
    TELEOP = 1
    TRACKING = 2
    SENTRY = 3

# --- GLOBAL VARIABLES ---
current_mode = AppMode.TELEOP
current_action = [0.0, 0.0]
recording = False
episode_path = None
csv_writer = None
csv_file = None
frame_count = 0

# --- ACTION BUFFER (The Lag Killer) ---
action_queue = collections.deque(maxlen=CHUNK_SIZE)
queue_lock = threading.Lock()

# Networking
control_session = requests.Session()
control_session.headers.update({"Connection": "keep-alive"})
CMD_KEEPALIVE = 0.10

# Shared video frame
latest_frame = None
frame_lock = threading.Lock()
capture_running = threading.Event()
sender_running = threading.Event()
desired_action = [0.0, 0.0]
udp_sock = None

# --- MOTOR KICKER (Fixes Deadband/Stalling) ---
MOTOR_CONFIG = {
    "left_trim": 1.0,
    "right_trim": 1.2,     # Boost right motor if it's lazy
    "min_power": 0.55,     # The "Kick" to overcome friction
    "stop_threshold": 0.1
}

def clean_motor_command(val, is_left_motor=True):
    trim = MOTOR_CONFIG["left_trim"] if is_left_motor else MOTOR_CONFIG["right_trim"]
    val = val * trim

    if abs(val) > MOTOR_CONFIG["stop_threshold"]:
        if val > 0:
            val = max(val, MOTOR_CONFIG["min_power"])
        else:
            val = min(val, -MOTOR_CONFIG["min_power"])
    else:
        val = 0.0

    return max(min(val, 1.0), -1.0)

# ---------------------------------------------------------------------------
# AI Model Definitions (MATCHING TRAIN SCRIPT)
# ---------------------------------------------------------------------------
class SentryMemoryPolicy(nn.Module):
    """
    Advanced CNN + LSTM Architecture.
    Matches 'train_advanced_memory.py'
    """
    def __init__(self):
        super().__init__()
        # 1. Vision Encoder (ResNet)
        resnet = resnet18(weights=None) # Weights loaded from file
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        
        # 2. Memory Cell (LSTM)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        
        # 3. Decision Head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, CHUNK_SIZE * 2) 
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 3, H, W)
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        features = self.encoder(c_in)
        features = features.view(batch_size, seq_len, 512)
        
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]
        
        return self.head(last_hidden)

class SentryBrain:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        # BUFFER for Temporal Memory
        self.frame_buffer = collections.deque(maxlen=SEQ_LEN)
        
        print(f"--- SENTRY BRAIN INITIALIZING ON {self.device.upper()} ---")
        
        if os.path.exists(SENTRY_MODEL_PATH):
            try:
                self.model = SentryMemoryPolicy()
                state_dict = torch.load(SENTRY_MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print(f"SUCCESS: Loaded Sentry MEMORY AI from {SENTRY_MODEL_PATH}")
            except Exception as e:
                print(f"ERROR: Failed to load Sentry AI: {e}")
                self.model = None
        else:
            print(f"WARNING: No model found at {SENTRY_MODEL_PATH}.")

    def decide(self, frame) -> List[Tuple[float, float]]:
        if self.model is None or frame is None:
            return []

        # 1. Preprocess Frame
        img = cv2.resize(frame, (240, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / \
            torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            
        # 2. Add to Memory Buffer
        self.frame_buffer.append(t)
        
        # 3. Only run if we have enough history
        if len(self.frame_buffer) < SEQ_LEN:
            return [] # Waiting for warm up...
            
        # 4. Create Sequence Batch
        # Stack to (Seq, C, H, W) -> Add Batch Dim -> (1, Seq, C, H, W)
        seq_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)

        # 5. Inference
        with torch.no_grad():
            output = self.model(seq_tensor) 
            
        flat = output.cpu().numpy().flatten()
        
        actions = []
        for i in range(0, len(flat), 2):
            v_left = float(flat[i])
            v_right = float(flat[i+1])
            # Raw output from model (usually clean, but we clip just in case)
            v_left = max(-1.0, min(1.0, v_left))
            v_right = max(-1.0, min(1.0, v_right))
            actions.append((v_left, v_right))
            
        return actions

# ---------------------------------------------------------------------------
# Hand Tracking (Standard)
# ---------------------------------------------------------------------------
class HandTracker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.present = False
        self.missing_frames = 0
        self.max_missing = 5
        self.landmarker = None
        if not enabled: return

        # Load MediaPipe
        def ensure_model():
            if not MODEL_PATH.exists():
                print("Downloading hand_landmarker.task ...")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
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
        except Exception as e:
            print(f"HandTracker Error: {e}")
            self.enabled = False

        self.prev = (0.0, 0.0)
        self.t0 = time.perf_counter()

    def infer(self, frame) -> Optional[Tuple[float, float]]:
        if not self.enabled or self.landmarker is None: return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - self.t0) * 1000)
        res = self.landmarker.detect_for_video(mp_image, ts_ms)
        
        if not res.hand_landmarks:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing: self.prev = (0.0, 0.0)
            return None
        self.missing_frames = 0
        
        wrist = res.hand_landmarks[0][0]
        # Simple Logic: Wrist Position controls drive
        forward = (0.5 - wrist.y) * 2.0
        turn = (0.5 - wrist.x) * 2.0
        
        l = forward - turn
        r = forward + turn
        return (l, r)

# ---------------------------------------------------------------------------
# Main Loop & Networking
# ---------------------------------------------------------------------------
try:
    import ctypes
    _GetKeyState = ctypes.windll.user32.GetAsyncKeyState
    _HAS_ASYNC = True
except:
    _HAS_ASYNC = False

def poll_drive_state():
    if not _HAS_ASYNC: return None
    pressed = lambda vk: (_GetKeyState(vk) & 0x8000) != 0
    if pressed(0x26) or pressed(ord('W')): return (1.0, 1.0)
    if pressed(0x28) or pressed(ord('S')): return (-1.0, -1.0)
    if pressed(0x25) or pressed(ord('A')): return (-0.5, 0.5)
    if pressed(0x27) or pressed(ord('D')): return (0.5, -0.5)
    return None

def send_command(left, right):
    try:
        url = f"{BASE_URL}/drive?l={left:.2f}&r={right:.2f}"
        control_session.get(url, timeout=0.08)
    except: pass

def sender_loop():
    global current_action
    while sender_running.is_set():
        start_time = time.time()
        
        if current_mode == AppMode.SENTRY:
            # Consume from Buffer
            with queue_lock:
                if len(action_queue) > 0:
                    tl, tr = action_queue.popleft()
                    current_action = [tl, tr] 
                else:
                    tl, tr = 0.0, 0.0
        else:
            tl, tr = current_action[0], current_action[1]

        # --- CRITICAL: APPLY MOTOR KICKER HERE ---
        final_l = clean_motor_command(tl, is_left_motor=True)
        final_r = clean_motor_command(tr, is_left_motor=False)
        
        try:
            if USE_UDP and udp_sock:
                payload = f"{final_l:.2f},{final_r:.2f}".encode()
                udp_sock.sendto(payload, (ESP32_IP, UDP_PORT))
            else:
                send_command(final_l, final_r)
        except: pass
        
        dt = time.time() - start_time
        if dt < FRAME_PERIOD: time.sleep(FRAME_PERIOD - dt)

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
    print(f"--- RECORDING: {episode_path} ---")

def stop_recording():
    global recording, csv_file, csv_writer
    if recording:
        recording = False
        if csv_file: csv_file.close()
        print(f"--- SAVED {frame_count} FRAMES ---")

def save_frame(frame, action):
    if not recording or not episode_path: return
    ts = int(time.time() * 1000)
    fn = f"{ts}.jpg"
    cv2.imwrite(os.path.join(episode_path, fn), frame)
    csv_writer.writerow([fn, action[0], action[1], ts])

def main():
    global current_action, current_mode, recording, udp_sock, latest_frame

    hand_tracker = HandTracker(ENABLE_HAND_TRACKING)
    sentry = SentryBrain()

    if USE_UDP:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setblocking(False)

    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("CAMERA ERROR")
        return

    capture_running.set()
    sender_running.set()

    def capture_loop():
        global latest_frame
        while capture_running.is_set():
            ok, frame = cap.read()
            if ok:
                with frame_lock: latest_frame = frame
            else: time.sleep(0.005)

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=sender_loop, daemon=True).start()

    print("\n--- CONTROLS: 1=Teleop, 2=Hand, 3=Sentry (LSTM), Space=Rec, Q=Quit ---")
    
    cv2.namedWindow('ESP32', cv2.WINDOW_NORMAL)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else blank

        if current_mode == AppMode.SENTRY:
            new_chunk = sentry.decide(frame)
            if new_chunk:
                with queue_lock:
                    # LSTM output is smooth, so we can just replace the queue
                    # or extend it. Extending is safer for continuous movement.
                    if len(action_queue) < 5: 
                         action_queue.extend(new_chunk)
            
            cv2.putText(frame, f"LSTM MEMORY: Active", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif current_mode == AppMode.TRACKING:
            ga = hand_tracker.infer(frame)
            if ga: current_action = list(ga)
            else: current_action = [0.0, 0.0]

        elif current_mode == AppMode.TELEOP:
             if recording:
                s_img = cv2.resize(frame, (224, 224))
                save_frame(s_img, current_action)

        # Draw HUD
        cv2.putText(frame, f"MODE: {current_mode.name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if recording: cv2.circle(frame, (300, 20), 5, (0, 0, 255), -1)

        cv2.imshow('ESP32', frame)
        key = cv2.waitKey(1)

        if key == ord('q'): break
        elif key == ord('1'): current_mode = AppMode.TELEOP
        elif key == ord('2'): current_mode = AppMode.TRACKING
        elif key == ord('3'): current_mode = AppMode.SENTRY
        elif key == 32: 
            if recording: stop_recording()
            else: start_recording()

        if current_mode == AppMode.TELEOP:
            ds = poll_drive_state()
            if ds: current_action = list(ds)
            elif key == -1: current_action = [0.0, 0.0]

    capture_running.clear()
    sender_running.clear()
    cap.release()
    cv2.destroyAllWindows()
    stop_recording()

if __name__ == '__main__':
    main()