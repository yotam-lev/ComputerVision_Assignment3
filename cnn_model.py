import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================================
# 0. HELPER & CONFIG
# ============================================================================

# === NEW ===
# Define file paths
VIDEO_ROOT = 'small-20bn-jester-v1'  # The directory containing your videos (e.g., 1, 2, 3...)
LABEL_FILE = 'jester-v1-labels.csv'
TRAIN_CSV = 'jester-v1-small-train.csv' # Use 'jester-v1-train.csv' for full training
VAL_CSV = 'jester-v1-validation.csv'

# === NEW ===
def load_label_map(label_file_path):
    """Loads the label CSV and returns a dictionary mapping string labels to integers."""
    labels_df = pd.read_csv(label_file_path, header=None, names=['label_str'])
    return {label_str: idx for idx, label_str in enumerate(labels_df['label_str'])}

# ============================================================================
# 1. DATASET CLASSES
# ============================================================================

# ============================================================================
# 1. DATASET CLASSES (CORRECTED FOR IMAGE DIRECTORIES)
# ============================================================================

class SingleFrameDataset(Dataset):
    """Baseline: Single random frame per video, from image directory"""
    
    # === CHANGED ===
    def __init__(self, annotation_file, root_dir, label_map, transform=None):
        """
        Args:
            annotation_file: Path to CSV with columns [id; label_str]
            root_dir: Path to the directory containing video *folders* (e.g., '1', '2', ...)
            label_map: Dictionary mapping string labels to integers
            transform: torchvision transforms to apply
        """
        self.data = pd.read_csv(annotation_file, sep=';', header=None, names=['id', 'label_str'])
        self.root_dir = Path(root_dir)
        self.label_map = label_map
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_id = self.data.iloc[idx, 0]
        label_str = self.data.iloc[idx, 1]
        label = self.label_map[label_str]
        
        # === NEW LOGIC: Read from image directory ===
        video_dir = self.root_dir / str(video_id)
        
        # Get list of all frame files
        frame_paths = sorted(list(video_dir.glob('*.jpg')))
        
        if not frame_paths:
            # Fallback: return black frame if directory is empty or missing
            print(f"Warning: No frames found in {video_dir}. Returning black frame.")
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Random frame selection
            frame_path = random.choice(frame_paths)
            
            # Load the single image frame
            frame = cv2.imread(str(frame_path))
            
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}. Returning black frame.")
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label


class VideoClipDataset(Dataset):
    """Improvement: Load video clips from image directories"""
    
    # === CHANGED ===
    def __init__(self, annotation_file, root_dir, label_map, transform=None, num_frames=16, 
                 use_optical_flow=False, img_size=224):
        """
        Args:
            annotation_file: Path to CSV with columns [id; label_str]
            root_dir: Path to the directory containing video *folders* (e.g., '1', '2', ...)
            label_map: Dictionary mapping string labels to integers
            transform: torchvision transforms
            num_frames: Number of frames per clip
            use_optical_flow: If True, compute optical flow instead of RGB
            img_size: Resize dimension
        """
        self.data = pd.read_csv(annotation_file, sep=';', header=None, names=['id', 'label_str'])
        self.root_dir = Path(root_dir)
        self.label_map = label_map
        self.transform = transform
        self.num_frames = num_frames
        self.use_optical_flow = use_optical_flow
        self.img_size = img_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_id = self.data.iloc[idx, 0]
        label_str = self.data.iloc[idx, 1]
        label = self.label_map[label_str]
        
        # === NEW LOGIC: Read from image directory ===
        video_dir = self.root_dir / str(video_id)
        
        # Get list of all frame files
        frame_paths = sorted(list(video_dir.glob('*.jpg')))
        total_frames = len(frame_paths)
        
        frames = []
        
        if total_frames == 0:
            print(f"Warning: No frames found in {video_dir}. Returning black frames.")
            for _ in range(self.num_frames + (1 if self.use_optical_flow else 0)):
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        else:
            # Determine starting frame
            num_to_sample = self.num_frames + (1 if self.use_optical_flow else 0)
            
            if total_frames > num_to_sample:
                start_frame = random.randint(0, total_frames - num_to_sample)
                selected_paths = frame_paths[start_frame : start_frame + num_to_sample]
            else:
                # Pad with last frame if video is too short
                selected_paths = frame_paths
                padding = [frame_paths[-1]] * (num_to_sample - total_frames)
                selected_paths.extend(padding)
            
            # Read frames from paths
            for frame_path in selected_paths:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    print(f"Warning: Could not read frame {frame_path}. Using black frame.")
                    frame = frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                frames.append(frame)
        
        # --- The rest of your logic is the same ---
        if self.use_optical_flow:
            return self._compute_optical_flow(frames), label
        else:
            return self._process_rgb_frames(frames[:-1] if len(frames) > self.num_frames else frames), label
    
    def _process_rgb_frames(self, frames):
        """Process RGB frames into tensor (C, T, H, W)"""
        processed = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            if self.transform:
                frame = self.transform(frame)
            processed.append(frame)
        
        # Stack: (T, C, H, W) -> (C, T, H, W)
        clip = torch.stack(processed, dim=1)
        return clip
    
    def _compute_optical_flow(self, frames):
        """Compute optical flow between consecutive frames"""
        flow_fields = []
        
        for i in range(len(frames) - 1):
            frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            
            frame1 = cv2.resize(frame1, (self.img_size, self.img_size))
            frame2 = cv2.resize(frame2, (self.img_size, self.img_size))
            
            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Normalize flow
            flow = (flow - flow.mean()) / (flow.std() + 1e-5)
            flow_fields.append(torch.from_numpy(flow).float())
        
        # Stack: (T, H, W, 2) -> (2, T, H, W)
        flow_tensor = torch.stack(flow_fields, dim=0).permute(3, 0, 1, 2)
        return flow_tensor
    """Improvement: Load video clips for temporal modeling"""
    
    # === CHANGED ===
    def __init__(self, annotation_file, root_dir, label_map, transform=None, num_frames=16, 
                 use_optical_flow=False, img_size=224):
        """
        Args:
            annotation_file: Path to CSV with columns [id; label_str]
            root_dir: Path to the directory containing video files
            label_map: Dictionary mapping string labels to integers
            transform: torchvision transforms
            num_frames: Number of frames per clip
            use_optical_flow: If True, compute optical flow instead of RGB
            img_size: Resize dimension
        """
        # === CHANGED ===
        self.data = pd.read_csv(annotation_file, sep=';', header=None, names=['id', 'label_str'])
        self.root_dir = Path(root_dir)
        self.label_map = label_map
        self.transform = transform
        self.num_frames = num_frames
        self.use_optical_flow = use_optical_flow
        self.img_size = img_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # === CHANGED ===
        video_id = self.data.iloc[idx, 0]
        label_str = self.data.iloc[idx, 1]
        label = self.label_map[label_str]
        
        video_path = str(self.root_dir / str(video_id))
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        
        if total_frames <= 0:
            print(f"Warning: Could not read video {video_path}. Returning black frames.")
            for _ in range(self.num_frames + (1 if self.use_optical_flow else 0)):
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        else:
            # Determine starting frame
            if total_frames > self.num_frames:
                start_frame = random.randint(0, total_frames - self.num_frames - 1)
            else:
                start_frame = 0
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read frames
            for _ in range(self.num_frames + (1 if self.use_optical_flow else 0)):
                ret, frame = cap.read()
                if not ret:
                    # Duplicate last frame if video ends
                    frame = frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                frames.append(frame)
        
        cap.release()
        
        if self.use_optical_flow:
            return self._compute_optical_flow(frames), label
        else:
            return self._process_rgb_frames(frames[:-1] if len(frames) > self.num_frames else frames), label
    
    def _process_rgb_frames(self, frames):
        """Process RGB frames into tensor (C, T, H, W)"""
        processed = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            if self.transform:
                frame = self.transform(frame)
            processed.append(frame)
        
        # Stack: (T, C, H, W) -> (C, T, H, W)
        clip = torch.stack(processed, dim=1)
        return clip
    
    def _compute_optical_flow(self, frames):
        """Compute optical flow between consecutive frames"""
        flow_fields = []
        
        for i in range(len(frames) - 1):
            frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            
            frame1 = cv2.resize(frame1, (self.img_size, self.img_size))
            frame2 = cv2.resize(frame2, (self.img_size, self.img_size))
            
            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Normalize flow
            flow = (flow - flow.mean()) / (flow.std() + 1e-5)
            flow_fields.append(torch.from_numpy(flow).float())
        
        # Stack: (T, H, W, 2) -> (2, T, H, W)
        flow_tensor = torch.stack(flow_fields, dim=0).permute(3, 0, 1, 2)
        return flow_tensor


# ============================================================================
# 2. MODEL ARCHITECTURES
# (No changes needed here)
# ============================================================================

class BaselineCNN(nn.Module):
    """Baseline: 2D CNN on single frames"""
    
    def __init__(self, num_classes=27, pretrained=True):
        super(BaselineCNN, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class CNN3D(nn.Module):
    """3D CNN for video clips"""
    
    def __init__(self, num_classes=27, pretrained=True):
        super(CNN3D, self).__init__()
        self.model = models.video.r3d_18(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class CNN_RNN(nn.Module):
    """2D CNN + LSTM for temporal modeling"""
    
    def __init__(self, num_classes=27, hidden_size=512, num_layers=1):
        super(CNN_RNN, self).__init__()
        
        # CNN feature extractor (remove classifier)
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_size = 512
        
        # LSTM
        self.lstm = nn.LSTM(self.feature_size, hidden_size, num_layers, batch_first=True)
        
        # Classifier
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        batch_size, channels, time_steps, h, w = x.size()
        
        # Reshape to process all frames: (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * time_steps, channels, h, w)
        
        # Extract features: (B*T, feature_size)
        features = self.cnn(x)
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM: (B, T, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use last hidden state
        out = self.fc(h_n[-1])
        return out


# ============================================================================
# 3. TRAINING LOOP
# (No changes needed here)
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=10, 
                learning_rate=0.001, device='cuda', save_path='best_model.pth'):
    """Training loop with validation"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_correct += predicted.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    return history


def evaluate_model(model, data_loader, criterion, device='cuda'):
    """Evaluate model on validation/test set"""
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    
    val_loss /= len(data_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


# ============================================================================
# 4. EVALUATION & VISUALIZATION
# (Changes to Grad-CAM target layer name for r3d_18)
# ============================================================================

def compute_confusion_matrix(model, data_loader, device='cuda', save_path='confusion_matrix.png'):
    """Compute and plot confusion matrix"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6) # Avoid division by zero
    
    return cm, per_class_acc, all_preds, all_targets


def gradcam_3d(model, input_tensor, target_class, device='cuda'):
    """Grad-CAM for 3D CNN"""
    model.eval()
    
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # === CHANGED ===
    # Target layer for r3d_18 is model.layer4[-1].conv2 (if it's a BasicBlock)
    # Let's target the final convolutional layer in layer4
    try:
        target_layer = model.model.layer4[-1].conv2
    except AttributeError:
        # Fallback for different block structure
        target_layer = model.model.layer4[-1].conv1
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    input_tensor = input_tensor.unsqueeze(0).to(device)
    output = model(input_tensor)
    
    # Backward pass
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()
    
    # Get activations and gradients
    act = activations[0].squeeze(0)  # (C, T, H, W)
    grad = gradients[0].squeeze(0)   # (C, T, H, W)
    
    # Compute weights
    weights = grad.mean(dim=(1, 2, 3), keepdim=True)  # (C, 1, 1, 1)
    
    # Weighted combination
    cam_3d = (weights * act).sum(dim=0)  # (T, H, W)
    cam_3d = torch.relu(cam_3d)
    
    # Average across time
    cam_2d = cam_3d.mean(dim=0)  # (H, W)
    
    # Normalize
    cam_2d = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + 1e-8)
    
    forward_handle.remove()
    backward_handle.remove()
    
    return cam_2d.cpu().detach().numpy()


def visualize_gradcam(frame, cam, save_path='gradcam.png', alpha=0.5):
    """Overlay Grad-CAM on frame"""
    if isinstance(frame, torch.Tensor):
        # Convert tensor to numpy: (C, H, W) -> (H, W, C)
        frame = frame.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed (assuming standard normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = std * frame + mean
        frame = np.clip(frame, 0, 1)
        frame = np.uint8(frame * 255)
    
    cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(frame, 1-alpha, heatmap, alpha, 0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title('Original Frame')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title('Grad-CAM Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    # Configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    num_classes = 27
    batch_size = 16
    num_epochs = 15 # Can be increased for better accuracy
    
    # === NEW ===
    # Load the label-to-integer mapping
    label_map = load_label_map(LABEL_FILE)
    
    # Transforms
    transform_single = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform for video clips (resize is handled in dataset)
    transform_video = transforms.Compose([
        transforms.ToPILImage(), # === This will fail if resize isn't done first
        # === Let's correct this. Resize is in _process_rgb_frames, but ToTensor needs PIL
        # === The original code is fine, ToPILImage works on numpy arrays.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("="*60)
    print("BASELINE: Single Frame 2D CNN")
    print(f"Training on: {TRAIN_CSV}, Validating on: {VAL_CSV}")
    print("="*60)
    
    # === CHANGED ===
    # Baseline dataset and model
    train_dataset = SingleFrameDataset(TRAIN_CSV, VIDEO_ROOT, label_map, transform=transform_single)
    val_dataset = SingleFrameDataset(VAL_CSV, VIDEO_ROOT, label_map, transform=transform_single)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    baseline_model = BaselineCNN(num_classes=num_classes)
    history_baseline = train_model(baseline_model, train_loader, val_loader, 
                                   num_epochs=num_epochs, device=device, 
                                   save_path='baseline_model.pth')
    
    # Evaluate baseline
    cm, per_class, preds, targets = compute_confusion_matrix(
        baseline_model, val_loader, device=device, save_path='baseline_cm.png'
    )
    
    print("\n" + "="*60)
    print("IMPROVEMENT: 3D CNN")
    print(f"Training on: {TRAIN_CSV}, Validating on: {VAL_CSV}")
    print("="*60)
    
    # === CHANGED ===
    # 3D CNN dataset and model
    train_dataset_3d = VideoClipDataset(TRAIN_CSV, VIDEO_ROOT, label_map, transform=transform_video, 
                                        num_frames=16, use_optical_flow=False)
    val_dataset_3d = VideoClipDataset(VAL_CSV, VIDEO_ROOT, label_map, transform=transform_video, 
                                      num_frames=16, use_optical_flow=False)
    
    # Reduce batch size for 3D model (VRAM intensive)
    train_loader_3d = DataLoader(train_dataset_3d, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_3d = DataLoader(val_dataset_3d, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    model_3d = CNN3D(num_classes=num_classes)
    history_3d = train_model(model_3d, train_loader_3d, val_loader_3d, 
                            num_epochs=num_epochs, device=device, 
                            save_path='3dcnn_model.pth')
    
    # Evaluate 3D CNN
    cm_3d, per_class_3d, preds_3d, targets_3d = compute_confusion_matrix(
        model_3d, val_loader_3d, device=device, save_path='3dcnn_cm.png'
    )
    
    print("\nTraining Complete!")
    print(f"Baseline Accuracy: {accuracy_score(targets, preds)*100:.2f}%")
    print(f"3D CNN Accuracy: {accuracy_score(targets_3d, preds_3d)*100:.2f}%")


if __name__ == '__main__':
    main()