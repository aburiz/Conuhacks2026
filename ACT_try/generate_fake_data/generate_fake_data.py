import cv2
import numpy as np
import pandas as pd
import os
import random

# CONFIG
NUM_EPISODES = 5
STEPS_PER_EPISODE = 50
SAVE_DIR = "dataset"

def generate_fake():
    print(f"Generating {NUM_EPISODES} fake episodes...")
    
    for ep in range(NUM_EPISODES):
        # Create folder: dataset/fake_episode_0
        ep_name = f"fake_episode_{ep}"
        ep_path = os.path.join(SAVE_DIR, ep_name)
        os.makedirs(ep_path, exist_ok=True)
        
        csv_data = []
        
        for i in range(STEPS_PER_EPISODE):
            # 1. Generate Random "Static" Image (Noise)
            # We assume 224x224 because that's what the model wants
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Make a "pattern" so the AI actually has something to learn 
            # (e.g., if i is high, make image brighter)
            img = img + (i * 2) 
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            filename = f"step_{i}.jpg"
            cv2.imwrite(os.path.join(ep_path, filename), img)
            
            # 2. Generate Random Actions (correlated with step 'i')
            # Let's pretend the robot drives faster as time goes on
            v_left = i / 100.0 
            v_right = i / 100.0
            
            csv_data.append([filename, v_left, v_right])
            
        # Save CSV
        df = pd.DataFrame(csv_data, columns=["filename", "v_left", "v_right"])
        df.to_csv(os.path.join(ep_path, "data.csv"), index=False)
        
    print(f"Done! Created {NUM_EPISODES * STEPS_PER_EPISODE} samples in '{SAVE_DIR}'")

if __name__ == "__main__":
    generate_fake()