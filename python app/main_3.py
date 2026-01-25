import cv2
import numpy as np
import urllib.request
import threading
import requests
import time
import os
import csv
from datetime import datetime
from enum import Enum

# --- CONFIGURATION ---
ESP32_IP = "192.168.0.129"  # Update if your IP changes
STREAM_URL = f"http://{ESP32_IP}:81/stream"
BASE_URL = f"http://{ESP32_IP}"
SAVE_DIR = "dataset"

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

def send_command(left, right):
    """Sends command to ESP32 non-blocking."""
    try:
        # Using a session with a short timeout to prevent lag
        url = f"{BASE_URL}/drive?l={left:.2f}&r={right:.2f}"
        control_session.get(url, timeout=0.1)
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
        frame_count += 1

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

def main():
    global current_action, current_mode, recording

    # Ensure dataset directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Connecting to stream at: {STREAM_URL}")
    try:
        stream = urllib.request.urlopen(STREAM_URL, timeout=5)
    except Exception as e:
        print(f"Error connecting to camera: {e}")
        return

    bytes_data = b''
    
    print("\n--- CONTROLS ---")
    print("TAB   : Switch Modes")
    print("SPACE : Toggle Recording (Teleop only)")
    print("WASD  : Drive")
    print("Q     : Quit")
    
    while True:
# --- 1. READ VIDEO STREAM ---
        try:
            bytes_data += stream.read(4096)
            
            # 1. Find the Start of Frame (0xff, 0xd8)
            a = bytes_data.find(b'\xff\xd8')
            
            if a != -1:
                # 2. Look for End of Frame (0xff, 0xd9) AFTER the start
                b = bytes_data.find(b'\xff\xd9', a)
                
                if b != -1:
                    # We have a complete frame!
                    jpg = bytes_data[a:b+2]
                    
                    # Advance buffer to the next potential frame
                    bytes_data = bytes_data[b+2:]
                    
                    # Safety check: ensure jpg is not empty before decoding
                    if len(jpg) > 0:
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        # Double check if decoding succeeded
                        if frame is not None:
                            
                            # --- 2. LOGIC BASED ON MODE ---
                            # (Paste your logic from before: if current_mode == ... etc)
                            
                            if current_mode == AppMode.TELEOP:
                                if recording:
                                    save_img = cv2.resize(frame, (224, 224))
                                    save_frame(save_img, current_action)
                                    threading.Thread(target=send_command, args=(current_action[0], current_action[1])).start()

                            elif current_mode == AppMode.TRACKING:
                                if current_action != [0.0, 0.0]:
                                    current_action = [0.0, 0.0]
                                    threading.Thread(target=send_command, args=(0, 0)).start()
                                cv2.putText(frame, "TRACKING NOT IMPLEMENTED", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            elif current_mode == AppMode.SENTRY:
                                cv2.putText(frame, "SENTRY MODE ACTIVE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            # --- 3. UI & DISPLAY ---
                            draw_hud(frame)
                            cv2.imshow('ESP32 RC Controller', frame)
            
            # Optional: prevent buffer from growing infinitely if no end tag is found for a long time
            if len(bytes_data) > 65536: # 64KB buffer limit
                bytes_data = b'' 
                
        except Exception as e:
            print(f"Stream Error: {e}")
            break

        # --- 4. INPUT HANDLING ---
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            stop_recording()
            break
            
        # Mode Switching (TAB key is usually 9)
        elif key == 9: 
            # Cycle modes
            new_mode_val = (current_mode.value % 3) + 1
            current_mode = AppMode(new_mode_val)
            # Reset state on mode switch
            stop_recording()
            current_action = [0.0, 0.0]
            send_command(0, 0)
            print(f"Switched to {current_mode.name}")

        # Teleop Controls
        if current_mode == AppMode.TELEOP:
            if key == 32: # SPACE to toggle record
                if recording:
                    stop_recording()
                else:
                    start_recording()
            
            # Movement Logic (Simple)
            if key == ord('w'): current_action = [1.0, 1.0]
            elif key == ord('s'): current_action = [-1.0, -1.0]
            elif key == ord('a'): current_action = [-0.5, 0.5] # Turn Left
            elif key == ord('d'): current_action = [0.5, -0.5] # Turn Right
            elif key == -1: 
                # If no key pressed, stop (Zero-order hold behavior)
                # Note: openCV waitKey returns -1 if no key. 
                # You might want 'holding' logic, but this stops immediately on release.
                current_action = [0.0, 0.0]
                
            # If you prefer "Hold key to drive" logic (safer):
            # The waitKey loop is fast, so resetting to 0.0 here works if keys aren't held.
        else:
            # Non-Teleop modes
            current_action = [0.0, 0.0]

    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()

if __name__ == '__main__':
    main()