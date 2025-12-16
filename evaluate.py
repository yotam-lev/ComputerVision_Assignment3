import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Local imports
from data.dataset import JesterDataset
from models.baseline import BaselineResNet18
from models.tsm import TSMResNet18


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total if total > 0 else 0.0
    return acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved checkpoint on validation set')
    parser.add_argument('--model', required=True, choices=['baseline', 'tsm'], help='Model type')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--csv_val', default='./datasets/jester-v1-validation.csv')
    parser.add_argument('--labels', default='./datasets/jester-v1-labels.csv')
    parser.add_argument('--image_root', default='./small-20bn-jester-v1')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Transforms (match train.py validation transforms)
    val_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset + loader according to model choice
    if args.model == 'baseline':
        n_segments = 1
        model = BaselineResNet18(num_classes=27).to(device)
    else:
        n_segments = 12
        model = TSMResNet18(num_classes=27, n_segments=n_segments).to(device)

    val_ds = JesterDataset(args.csv_val, args.image_root, args.labels, val_tx, n_segments=n_segments)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load checkpoint
    state = torch.load(args.checkpoint, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # If checkpoint is a dict with key 'state_dict' or similar, try common keys
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            # last resort: attempt to load keys that match
            model.load_state_dict(state)

    acc = validate(model, val_loader, device)
    print(f"✅ Validation Accuracy: {acc:.2f}%")


if __name__ == '__main__':
    main()
