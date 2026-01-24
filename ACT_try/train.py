import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import glob
import os
import numpy as np

# --- CONFIG ---
CHUNK_SIZE = 10 # Predict 10 steps into future
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

class RobotDataset(Dataset):
    def __init__(self, root_dir):
        self.frames = []
        self.actions = []
        
        # Load all CSVs from all episode folders
        for csv_file in glob.glob(f"{root_dir}/*/data.csv"):
            df = pd.read_csv(csv_file)
            folder = os.path.dirname(csv_file)
            
            # Create chunks
            for i in range(len(df) - CHUNK_SIZE):
                img_path = os.path.join(folder, df.iloc[i]['filename'])
                
                # Get next CHUNK_SIZE actions
                # Shape: (10, 2) -> flattened to (20,)
                act_chunk = df.iloc[i:i+CHUNK_SIZE][['v_left', 'v_right']].values.flatten()
                
                self.frames.append(img_path)
                self.actions.append(act_chunk)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Load Image
        img = cv2.imread(self.frames[idx])
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0 # (3, 224, 224)
        
        # Load Action
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        
        return img, action

class SentryPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet18 for vision
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity() # Remove last layer
        
        # Action Head: 512 features -> 20 outputs (10 steps * 2 motors)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, CHUNK_SIZE * 2) 
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# --- TRAINING LOOP ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    dataset = RobotDataset("dataset")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SentryPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Simple regression loss
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, acts in loader:
            imgs, acts = imgs.to(device), acts.to(device)
            
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, acts)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "sentry_policy.pth")
    print("Model Saved!")