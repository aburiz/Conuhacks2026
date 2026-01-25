import cv2
import pandas as pd
import os
import time

# --- CONFIGURATION ---
# Paste your folder path here
EPISODE_PATH = r"dataset/20260125_034109" 
PLAYBACK_SPEED = 30  # FPS (Adjust if it's too fast/slow)

def draw_hud(frame, v_left, v_right, frame_idx, total_frames):
    """Overlays action bars and progress on the frame."""
    h, w, _ = frame.shape
    
    # 1. Progress Info
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 2. Left Motor Bar
    l_val = int(v_left * 50)
    cv2.rectangle(frame, (10, h - 20), (10 + 100, h - 10), (50, 50, 50), -1) # Background
    color_l = (0, 255, 0) if v_left > 0 else (0, 0, 255) # Green fwd, Red back
    cv2.rectangle(frame, (60, h - 20), (60 + l_val, h - 10), color_l, -1) # Bar
    
    # 3. Right Motor Bar
    r_val = int(v_right * 50)
    cv2.rectangle(frame, (120, h - 20), (220, h - 10), (50, 50, 50), -1) # Background
    color_r = (0, 255, 0) if v_right > 0 else (0, 0, 255)
    cv2.rectangle(frame, (170, h - 20), (170 + r_val, h - 10), color_r, -1) # Bar

    # Text Values
    cv2.putText(frame, f"L: {v_left:.2f}", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"R: {v_right:.2f}", (120, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
    csv_path = os.path.join(EPISODE_PATH, "data.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find data.csv in {EPISODE_PATH}")
        return

    # Load Data
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip() # Fix whitespace issues if any
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"--- REPLAYING: {EPISODE_PATH} ---")
    print(f"Total Frames: {len(df)}")
    print("Press 'Q' to quit early.")

    for i, row in df.iterrows():
        # Get paths and values
        img_name = row['filename']
        v_left = row['v_left']
        v_right = row['v_right']
        
        img_full_path = os.path.join(EPISODE_PATH, img_name)
        
        # Load Image
        frame = cv2.imread(img_full_path)
        
        if frame is None:
            print(f"Warning: Missing image {img_name}")
            continue
            
        # Draw Interface
        draw_hud(frame, v_left, v_right, i, len(df))
        
        # Show
        cv2.imshow("Dataset Replay", frame)
        
        # Handle Quit
        if cv2.waitKey(int(1000/PLAYBACK_SPEED)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Replay finished.")

if __name__ == "__main__":
    main()