import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Import your custom modules
from data.dataset import JesterDataset
from models.baseline import BaselineResNet18
from models.tsm import TSMResNet18

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(loader):
        # Move data to GPU/CPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print every 20 batches
        if i % 20 == 0:
            print(f"   Batch {i}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    # --- 1. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'tsm'], 
                        help='Choose "baseline" (2D CNN) or "tsm" (Temporal Shift)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    # --- 2. CONFIGURATION & PATHS ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Paths (Ensure these match your folder structure exactly)
    # Using os.path.join for safety, but hardcoded strings work if folders exist
    CSV_PATH_TRAIN = './datasets/jester-v1-small-train.csv'
    CSV_PATH_VAL   = './datasets/jester-v1-validation.csv'
    LABELS_PATH    = './datasets/jester-v1-labels.csv'
    IMAGE_ROOT     = './small-20bn-jester-v1' 

    # Verify paths exist before starting
    if not os.path.exists(CSV_PATH_TRAIN) or not os.path.exists(IMAGE_ROOT):
        raise FileNotFoundError("CRITICAL: Check your CSV_PATH or IMAGE_ROOT in train.py")

    # --- 3. TRANSFORMS ---
    # Standard ResNet Normalization
    #Training: Add randomness (Scale & Crop) to fight overfitting
    train_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    # --- 4. INITIALIZE DATASETS & MODELS ---
    print(f"🏗️  Initializing {args.model.upper()} model...")

    if args.model == 'baseline':
        # BASELINE: Sample 1 frame, standard 2D ResNet
        train_ds = JesterDataset(CSV_PATH_TRAIN, IMAGE_ROOT, LABELS_PATH, train_tx, n_segments=1)
        val_ds   = JesterDataset(CSV_PATH_VAL,   IMAGE_ROOT, LABELS_PATH, val_tx, n_segments=1)
        
        # Initialize Baseline Model
        model = BaselineResNet18(num_classes=27).to(device)

    elif args.model == 'tsm':
        # TSM: Sample 8 frames, TSM-injected ResNet
        train_ds = JesterDataset(CSV_PATH_TRAIN, IMAGE_ROOT, LABELS_PATH, train_tx, n_segments=12)
        val_ds   = JesterDataset(CSV_PATH_VAL,   IMAGE_ROOT, LABELS_PATH, val_tx, n_segments=12)
        
        # Initialize TSM Model
        model = TSMResNet18(num_classes=27, n_segments=12).to(device)

    # --- 5. DATA LOADERS ---
    # num_workers=2 is usually safe on Mac. If it crashes, set to 0.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"✅ Data Loaded. Training on {len(train_ds)} samples, Validating on {len(val_ds)} samples.")

    # --- 6. OPTIMIZER & LOSS ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # Optional: Scheduler to lower LR if training stalls
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 7. TRAINING LOOP ---
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\n📢 Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        print("   Validating...")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step()

        # Print Epoch Summary
        print(f"📊 Result: Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        print(f"           Val Loss:   {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f"{args.model}_v2_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved new best model to {save_path} (Acc: {best_acc:.2f}%)")

    print("\n🏁 Training Complete.")

# --- CRITICAL FOR MACOS MULTIPROCESSING ---
if __name__ == '__main__':
    main()