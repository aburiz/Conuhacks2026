import torch
import cv2
import numpy as np
import glob
import os
import time
from train_ResNet_18_architecture_ColorJitter import SentryPolicy, CONFIG # Import your model class

# --- CONFIG ---
MODEL_PATH = "output/sentry_policy_LATEST.pth" # <--- PUT YOUR MODEL FILE HERE
DATASET_DIR = "dataset"

def draw_bars(frame, human_left, human_right, ai_left, ai_right):
    """Draws visual bars to compare Human vs AI motor commands"""
    h, w, _ = frame.shape
    center_y = h // 2
    
    # Scale factors
    bar_width = 20
    scale = 100 # How long the bar gets
    
    # Background for bars
    cv2.rectangle(frame, (10, center_y - 100), (30, center_y + 100), (50, 50, 50), -1) # Left BG
    cv2.rectangle(frame, (w - 30, center_y - 100), (w - 10, center_y + 100), (50, 50, 50), -1) # Right BG
    
    # --- HUMAN (GREEN) ---
    # Left Motor
    h_l_height = int(human_left * scale)
    cv2.rectangle(frame, (12, center_y), (28, center_y - h_l_height), (0, 255, 0), -1)
    # Right Motor
    h_r_height = int(human_right * scale)
    cv2.rectangle(frame, (w - 28, center_y), (w - 12, center_y - h_r_height), (0, 255, 0), -1)

    # --- AI (RED) - Drawn slightly narrower on top ---
    # Left Motor
    ai_l_height = int(ai_left * scale)
    cv2.rectangle(frame, (15, center_y), (25, center_y - ai_l_height), (0, 0, 255), -1)
    # Right Motor
    ai_r_height = int(ai_right * scale)
    cv2.rectangle(frame, (w - 25, center_y), (w - 15, center_y - ai_r_height), (0, 0, 255), -1)
    
    # Labels
    cv2.putText(frame, "L", (10, center_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "R", (w-30, center_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Green=Human", (w//2 - 50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "Red=AI", (w//2 - 50, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    model = SentryPolicy().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model: {MODEL_PATH}")
    except:
        print(f"Could not load {MODEL_PATH}. Check filename.")
        return

    model.eval()
    
    # 2. Find Episodes
    episodes = glob.glob(f"{DATASET_DIR}/*")
    print(f"Found {len(episodes)} episodes to review.")
    
    for ep in episodes:
        csv_path = os.path.join(ep, "data.csv")
        if not os.path.exists(csv_path): continue
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        print(f"Replaying {ep}...")
        
        for i in range(len(df)):
            # Load Frame
            img_name = df.iloc[i]['filename']
            img_path = os.path.join(ep, img_name)
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # Prepare for AI (Resize -> Tensor -> Normalize)
            ai_input = cv2.resize(frame, (240, 240))
            ai_tensor = torch.tensor(ai_input, dtype=torch.float32).permute(2, 0, 1) / 255.0
            ai_tensor = ai_tensor.unsqueeze(0).to(device) # Add batch dim
            
            # Get AI Prediction
            with torch.no_grad():
                pred_chunk = model(ai_tensor).cpu().numpy().flatten()
                # We visualize just the immediate next step (Step 0)
                ai_left = pred_chunk[0]
                ai_right = pred_chunk[1]
                
            # Get Human Ground Truth
            human_left = df.iloc[i]['v_left']
            human_right = df.iloc[i]['v_right']
            
            # Draw Visualization
            display_frame = cv2.resize(frame, (480, 480)) # Make it big to see
            display_frame = draw_bars(display_frame, human_left, human_right, ai_left, ai_right)
            
            cv2.imshow("Policy Evaluation", display_frame)
            
            # Playback speed (wait 33ms = ~30fps)
            if cv2.waitKey(33) == ord('q'):
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluate()