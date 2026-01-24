import cv2
import numpy as np
import urllib.request
import threading
import requests

# --- CONFIGURATION ---
# The IP you see in your browser
ESP32_IP = "192.168.0.129"

# The standard CameraWebServer usually puts the stream on Port 81
STREAM_URL = f"http://{ESP32_IP}:81/stream"

# Control URL (We will build this next, it lives on Port 80)
BASE_URL = f"http://{ESP32_IP}"
def get_video_stream():
    """
    Connects to the stream and yields frames.
    Fixed: Ensures we only look for an End marker AFTER the Start marker.
    """
    try:
        # Connect to stream
        stream = urllib.request.urlopen(STREAM_URL, timeout=5)
    except Exception as e:
        print(f"Error connecting to stream: {e}")
        return

    bytes_data = b''
    
    while True:
        # Read a larger chunk to ensure we get full frames faster
        try:
            bytes_data += stream.read(4096)
            
            # 1. Find the Start of the JPEG (0xff 0xd8)
            a = bytes_data.find(b'\xff\xd8')
            
            # If no Start, discard garbage before it (optional) or just wait for more data
            if a == -1:
                continue 
            
            # 2. Find the End of the JPEG (0xff 0xd9) *starting from* where we found the Start
            b = bytes_data.find(b'\xff\xd9', a)
            
            # If we found a Start but no End, we need to read more data
            if b == -1:
                continue
            
            # 3. We have a complete frame from a to b
            jpg = bytes_data[a:b+2]
            
            # 4. Remove the processed frame from the buffer; keep the rest
            bytes_data = bytes_data[b+2:]
            
            # Decode
            if len(jpg) > 0:
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
                    
        except Exception as e:
            print(f"Stream loop error: {e}")
            break

def main():
    print(f"Attempting to connect to {STREAM_URL}...")
    
    for frame in get_video_stream():
        # --- VISION PROCESSING SECTION ---
        # (This is where we will add the Sentry/Tracker logic later)
        
        # Example: Flip it if the camera is mounted upside down
        # frame = cv2.flip(frame, 0) 
        
        # Display the frame
        cv2.imshow('Sentry Bot View', frame)
        
        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
            
        # Placeholder for controls (we will add the requests here next)
        elif key == ord('w'):
            print("Sending: FORWARD") 
            # requests.get(f"{BASE_URL}/action?go=forward")
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()