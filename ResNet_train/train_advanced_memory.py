import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
import cv2
import glob
import os
import numpy as np
from datetime import datetime
import wandb

# --- ADVANCED CONFIG ---
CONFIG = {
    "chunk_size": 20,       # Predict 20 steps into the future
    "seq_len": 5,           # MEMORY: Look at 5 past frames to make a decision
    "batch_size": 64,       # Slightly smaller batch because LSTM uses more VRAM
    "epochs": 50,           # More epochs because LSTM takes longer to learn
    "lr": 1e-4,
    "architecture": "ResNet18_LSTM",
    "output_dir": "output/advanced"
}

# --- DATASET WITH SEQUENCE LOADING ---
class RobotMemoryDataset(Dataset):
    def __init__(self, root_dir, seq_len=5):
        self.samples = [] # List of (image_path_list, action_chunk)
        self.seq_len = seq_len
        self.chunk_size = CONFIG["chunk_size"]
        
        # Robust Transforms
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        files = glob.glob(f"{root_dir}/*/data.csv")
        print(f"Loading {len(files)} episodes for Temporal Training...")

        for csv_file in files:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            folder = os.path.dirname(csv_file)
            
            # Need enough data for Past (seq_len) AND Future (chunk_size)
            if len(df) < (self.seq_len + self.chunk_size): continue

            # Create sliding windows
            for i in range(len(df) - self.chunk_size - self.seq_len):
                # 1. Grab sequence of PAST images
                img_seq = []
                for j in range(self.seq_len):
                    fname = df.iloc[i + j]['filename']
                    img_seq.append(os.path.join(folder, fname))
                
                # 2. Grab FUTURE action chunk (starting after the sequence)
                start_act = i + self.seq_len
                act_chunk = df.iloc[start_act : start_act + self.chunk_size][['v_left', 'v_right']].values.flatten()
                
                self.samples.append((img_seq, act_chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, action_raw = self.samples[idx]
        
        # Load all images in the sequence
        tensors = []
        for p in img_paths:
            img = cv2.imread(p)
            if img is None: img = np.zeros((240, 240, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (240, 240))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to Tensor (3, 240, 240)
            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            t = self.transform(t) # Apply jitter to each frame independently
            tensors.append(t)
            
        # Stack into (Seq_Len, 3, 240, 240)
        seq_tensor = torch.stack(tensors) 
        action = torch.tensor(action_raw, dtype=torch.float32)
        
        return seq_tensor, action

# --- ADVANCED MODEL: CNN + LSTM ---
class SentryMemoryPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        
        # 1. Vision Encoder (ResNet)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove the last classification layer, we want the 512 features
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        
        # 2. Memory Cell (LSTM)
        # Input: 512 features from ResNet
        # Hidden: 256 memory units
        # Layers: 2 stacked LSTMs for complex reasoning
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        
        # 3. Decision Head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, CONFIG["chunk_size"] * 2) 
        )

    def forward(self, x):
        # Input shape: (Batch, Seq_Len, 3, H, W)
        batch_size, seq_len, C, H, W = x.size()
        
        # Flatten sequence to feed into CNN: (Batch*Seq_Len, 3, H, W)
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        # Extract features
        features = self.encoder(c_in) # Output: (Batch*Seq, 512, 1, 1)
        features = features.view(batch_size, seq_len, 512) # Reshape back to sequence
        
        # Run Memory (LSTM)
        # We only care about the LAST output (the decision based on the whole history)
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :] # Take the last step
        
        return self.head(last_hidden)

if __name__ == "__main__":
    wandb.init(project="ConUhacks-Advanced", config=CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Training Advanced Memory Model on {device} ---")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 1. Load Data
    full_dataset = RobotMemoryDataset("dataset", seq_len=CONFIG["seq_len"])
    
    # 2. Split Train/Val (90/10) - Crucial for avoiding overfitting
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train Samples: {len(train_data)} | Val Samples: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    model = SentryMemoryPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()
    
    # Scheduler: Reduce LR if validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val = float('inf')

    for epoch in range(CONFIG["epochs"]):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for seq, act in train_loader:
            seq, act = seq.to(device), act.to(device)
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, act)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, act in val_loader:
                seq, act = seq.to(device), act.to(device)
                pred = model(seq)
                val_loss += criterion(pred, act).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        # Step the scheduler
        scheduler.step(avg_val)
        
        print(f"Epoch {epoch+1}: Train {avg_train:.4f} | Val {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        wandb.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch+1})
        
        # Save Best
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), f"{CONFIG['output_dir']}/best_memory_model.pth")
            print("  >>> Saved new best model")

    wandb.finish()