import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from models.ops import TSM

class TSMResNet18(nn.Module):
    def __init__(self, num_classes=27, n_segments=8):
        super().__init__()
        self.n_segments = n_segments
        
        # Load Backbone
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify Classifier
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)
        
        # Initialize TSM Module
        self.tsm = TSM(n_segment=n_segments, fold_div=8)

        # HACK: Manually wrapping the forward pass of ResNet 
        # to insert TSM before specific layers.
        # Ideally, you'd patch the blocks, but this is clearer for the assignment.

    def forward(self, x):
        # Input: [Batch, Segments, 3, 224, 224]
        b, s, c, h, w = x.size()
        
        # Flatten: [Batch * Segments, 3, 224, 224]
        x = x.view(b * s, c, h, w)

        # --- Standard ResNet Stem ---
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        # --- Layer 1 (Usually no shift needed here, but can add if desired) ---
        x = self.base_model.layer1(x)

        # --- Layer 2 (Inject TSM) ---
        x = self.tsm(x)
        x = self.base_model.layer2(x)

        # --- Layer 3 (Inject TSM) ---
        x = self.tsm(x)
        x = self.base_model.layer3(x)

        # --- Layer 4 (Inject TSM) ---
        x = self.tsm(x)
        x = self.base_model.layer4(x)

        # --- Head ---
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)

        # --- Consensus (Averaging) ---
        # Reshape back to [Batch, Segments, Classes]
        x = x.view(b, s, -1)
        
        # Average predictions across all 8 frames
        return x.mean(dim=1)
