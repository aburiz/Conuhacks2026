import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import cv2
import glob
import os
import numpy as np
from datetime import datetime
import wandb

# --- CONFIG ---
CONFIG = {
    "chunk_size": 20,
    "batch_size": 32,
    "epochs": 25,
    "lr": 1e-4,
    "architecture": "ResNet18_Temporal_Stack2", # Stacking 2 frames
    "output_dir": "output/temporal"
}

class TemporalRobotDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.chunk_size = CONFIG["chunk_size"]
        
        # 1. SPLIT TRANSFORMS
        # Augmentation (Applied to 3-channel images individually)
        self.augment = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        
        # Normalization (Applied to the final 6-channel stack)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], # Double the mean
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]   # Double the std
        )
        
        files = glob.glob(f"{root_dir}/*/data.csv")
        if not files:
            print(f"WARNING: No data found in {root_dir}!")

        for csv_file in files:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            folder = os.path.dirname(csv_file)
            
            # Start at index 1 so we always have a "previous" frame
            for i in range(1, len(df) - self.chunk_size):
                curr_img = os.path.join(folder, df.iloc[i]['filename'])
                prev_img = os.path.join(folder, df.iloc[i-1]['filename']) # Previous frame
                
                act_chunk = df.iloc[i:i+self.chunk_size][['v_left', 'v_right']].values.flatten()
                
                self.samples.append((prev_img, curr_img, act_chunk))
        
        print(f"Dataset Loaded: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prev_path, curr_path, actions = self.samples[idx]
        
        # Load Images (H, W, 3)
        img_prev = self._load_img(prev_path)
        img_curr = self._load_img(curr_path)
        
        # Convert to Tensor (3, H, W) and scale to 0-1
        t_prev = torch.tensor(img_prev, dtype=torch.float32).permute(2, 0, 1) / 255.0
        t_curr = torch.tensor(img_curr, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # --- APPLY AUGMENTATION INDIVIDUALLY ---
        # Note: Ideally we want the exact same random jitter for both frames 
        # to preserve temporal consistency, but independent jitter is also 
        # a valid (and stronger) regularization strategy.
        t_prev = self.augment(t_prev)
        t_curr = self.augment(t_curr)
        
        # --- STACK (6, H, W) ---
        stacked = torch.cat([t_prev, t_curr], dim=0)
        
        # --- NORMALIZE (Works on 6 channels) ---
        stacked = self.normalize(stacked)
        
        return stacked, torch.tensor(actions, dtype=torch.float32)

    def _load_img(self, path):
        img = cv2.imread(path)
        if img is None: return np.zeros((240, 240, 3), dtype=np.uint8)
        img = cv2.resize(img, (240, 240))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class TemporalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # ResNet expects 3 channels. We have 6.
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=6, 
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )
        
        # Initialize new weights
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = old_conv.weight
            self.backbone.conv1.weight[:, 3:] = old_conv.weight

        self.backbone.fc = nn.Identity() 
        
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG["chunk_size"] * 2),
            # nn.Tanh() 
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# --- TRAINING LOOP ---
if __name__ == "__main__":
    wandb.init(
        project="ConUhacks",
        entity="emath-mrl",
        config=CONFIG
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    dataset = TemporalRobotDataset("dataset")
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        wandb.finish()
        exit()
        
    # loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    # New (Optimized)
    loader = DataLoader(
        dataset, 
        batch_size=128,          # Try 128 or 256. 32 is too small.
        shuffle=True, 
        num_workers=16,           # Critical: Uses 8 CPU cores to prep images ahead of time
        pin_memory=True,         # Faster CPU -> GPU transfer
        persistent_workers=True  # Keeps workers alive between epochs
    )
    model = TemporalPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss() 
    
    wandb.watch(model, log="all")

    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        for imgs, acts in loader:
            imgs, acts = imgs.to(device), acts.to(device)
            
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, acts)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "learning_rate": CONFIG["lr"]
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(CONFIG["output_dir"], f"sentry_policy_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model Saved to: {save_path}")
    
    artifact = wandb.Artifact(f'model-{timestamp}', type='model')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()