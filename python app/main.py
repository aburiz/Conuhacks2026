import cv2
import urllib.request
import numpy as np
import requests
import threading

# CONFIGURATION
# Replace with the IP address your ESP32 gets from the Hotspot
ESP32_IP = "192.168.137.100" 
STREAM_URL = f"http://{ESP32_IP}:81/stream"
CONTROL_URL = f"http://{ESP32_IP}/action"

def send_command(action):
    """Sends a non-blocking request to the ESP32"""
    try:
        # Example: http://192.168.1.10/action?go=forward
        requests.get(f"{CONTROL_URL}?go={action}", timeout=0.1)
    except:
        pass

def remote_control_loop():
    # Open the video stream
    stream = urllib.request.urlopen(STREAM_URL)
    bytes_data = b''

    print("--- Remote Control Active ---")
    print("Use WASD to move. Press 'q' to quit.")

    while True:
        # 1. Read the MJPEG Stream
        bytes_data += stream.read(1024)
        a = bytes_data.find(b'\xff\xd8') # JPEG Start
        b = bytes_data.find(b'\xff\xd9') # JPEG End

        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            
            # Decode image
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            # --- THIS IS WHERE WE WILL ADD AI LATER ---
            # if mode == "sentry": frame = detect_danger(frame)
            # if mode == "follower": frame = track_hand(frame)
            
            cv2.imshow('Sentry Bot Feed', frame)

        # 2. Control Logic
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('w'):
            threading.Thread(target=send_command, args=("forward",)).start()
        elif key == ord('s'):
            threading.Thread(target=send_command, args=("backward",)).start()
        elif key == ord('a'):
            threading.Thread(target=send_command, args=("left",)).start()
        elif key == ord('d'):
            threading.Thread(target=send_command, args=("right",)).start()
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    remote_control_loop()