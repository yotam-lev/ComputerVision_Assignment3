import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaselineResNet18(nn.Module):
    """
    A standard 2D CNN Baseline.
    It processes a single frame independently, ignoring temporal information.
    """
    def __init__(self, num_classes=27):
        super().__init__()
        
        # 1. Load Pre-trained Weights (Transfer Learning)
        # We use IMAGENET1K_V1 to give the model a 'head start' on visual features
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. Modify the Classifier
        # ResNet18's final layer is named 'fc'. We replace it to match Jester's 27 classes.
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # 3. Input Handling
        # The DataLoader might pass [Batch, 1, 3, 224, 224] if n_segments=1.
        # Standard ResNet expects [Batch, 3, 224, 224].
        if x.dim() == 5:
            # Remove the temporal dimension (index 1)
            x = x.squeeze(1)
            
        # 4. Forward Pass
        return self.backbone(x)