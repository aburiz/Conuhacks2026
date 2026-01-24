import cv2
import numpy as np
import urllib.request
import threading
import requests
import time
import os
import csv
from datetime import datetime

# --- CONFIG ---
ESP32_IP = "192.168.0.129"
STREAM_URL = f"http://{ESP32_IP}:81/stream"
BASE_URL = f"http://{ESP32_IP}"
SAVE_DIR = "dataset"

# Create unique episode folder
episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
episode_path = os.path.join(SAVE_DIR, episode_id)
os.makedirs(episode_path, exist_ok=True)
print(f"Recording to: {episode_path}")

# Global state
current_action = [0.0, 0.0] # [left_vel, right_vel]
recording = False

def send_command(left, right):
    # Send to ESP32 (optimize this to be non-blocking/UDP later if needed)
    try:
        requests.get(f"{BASE_URL}/action?left={left}&right={right}", timeout=0.05)
    except:
        pass

def save_data(frame, action):
    """Saves one frame and its corresponding action"""
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}.jpg"
    
    # Save Image
    cv2.imwrite(os.path.join(episode_path, filename), frame)
    
    # Append to CSV
    with open(os.path.join(episode_path, "data.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        # format: filename, left_vel, right_vel
        writer.writerow([filename, action[0], action[1]])

def main():
    global current_action, recording
    
    # Setup CSV header
    with open(os.path.join(episode_path, "data.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "v_left", "v_right"])

    stream = urllib.request.urlopen(STREAM_URL)
    bytes_data = b''

    print("--- READY ---")
    print("HOLD 'SPACE' to RECORD while driving.")
    
    while True:
        # 1. Image Capture Logic (Same as before)
        bytes_data += stream.read(4096)
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # Resize to match model input (saves space/time)
            # 224x224 is standard for ResNet
            frame_resized = cv2.resize(frame, (224, 224))
            
            # 2. Record if Spacebar is held
            if recording:
                save_data(frame_resized, current_action)
                cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1) # Red dot

            cv2.imshow('Recorder', frame)

        # 3. Controls (WASD -> Velocity)
        keys = cv2.waitKey(1)
        if keys == ord('q'): break
        
        # Simple Logic: W=(1,1), A=(-1,1), S=(-1,-1), D=(1,-1)
        if keys == ord('w'): current_action = [1.0, 1.0]
        elif keys == ord('s'): current_action = [-1.0, -1.0]
        elif keys == ord('a'): current_action = [-0.5, 0.5]
        elif keys == ord('d'): current_action = [0.5, -0.5]
        elif keys == 32: # SPACE
             recording = True
             continue # Don't reset action
        else:
            current_action = [0.0, 0.0]
            recording = False # Stop recording if no keys
            
        # Send to robot
        if recording: # Only move if we are "driving/recording"
             threading.Thread(target=send_command, args=(current_action[0], current_action[1])).start()

if __name__ == '__main__':
    main()