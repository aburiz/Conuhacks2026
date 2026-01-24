import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import glob
import os
import numpy as np
from datetime import datetime
import wandb  # <--- IMPORT WANDB

# --- CONFIG ---
CONFIG = {
    "chunk_size": 10,
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-4,
    "architecture": "ResNet18_Modified",
    "output_dir": "output"
}

class RobotDataset(Dataset):
    def __init__(self, root_dir):
        self.frames = []
        self.actions = []
        
        files = glob.glob(f"{root_dir}/*/data.csv")
        if not files:
            print(f"WARNING: No data found in {root_dir}!")
            
        for csv_file in files:
            df = pd.read_csv(csv_file)
            folder = os.path.dirname(csv_file)
            
            for i in range(len(df) - CONFIG["chunk_size"]):
                img_path = os.path.join(folder, df.iloc[i]['filename'])
                act_chunk = df.iloc[i:i+CONFIG["chunk_size"]][['v_left', 'v_right']].values.flatten()
                self.frames.append(img_path)
                self.actions.append(act_chunk)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = cv2.imread(self.frames[idx])
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0 
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        return img, action

class SentryPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity() 
        
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG["chunk_size"] * 2) 
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# --- TRAINING LOOP ---
if __name__ == "__main__":
    # 1. Initialize WandB
    wandb.init(
        project="ConUhacks",
        entity="emath-mrl",
        config=CONFIG
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    dataset = RobotDataset("dataset")
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        wandb.finish() # Clean exit
        exit()
        
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    model = SentryPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss() 
    
    # Optional: Watch the model to see gradients in dashboard
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
        
        # 2. Log Metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "learning_rate": CONFIG["lr"]
        })
    
    # Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(CONFIG["output_dir"], f"sentry_policy_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model Saved to: {save_path}")
    
    # 3. Save model as an artifact to WandB (Optional but cool)
    artifact = wandb.Artifact(f'model-{timestamp}', type='model')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()