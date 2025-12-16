import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class TSM(nn.Module):
    """
    Standard Temporal Shift Module (TSM).
    Shifts 1/8 of channels to the past, 1/8 to the future, and leaves the rest.
    """
    def __init__(self, n_segment=12, fold_div=8):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        
        # Shift Left (Past)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # Shift Right (Future)
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        # Remainder (No Shift)
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        
        return out.view(nt, c, h, w)

class TIPS(nn.Module):
    """
    Translation Invariant Polyphase Sampling (TIPS).
    Replaces standard pooling to make the model robust to small spatial shifts.
    """
    def __init__(self, in_channels, stride=2):
        super(TIPS, self).__init__()
        self.stride = stride
        # FIXED: Changed '==' to '=' below
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, stride * stride, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self._init_weights()

    def _init_weights(self):
        # Initialize the last conv layer to 0 so it starts with even weighting
        nn.init.constant(self.weight_gen[3].weight, 0)
        nn.init.constant(self.weight_gen[3].bias, 0)

    def forward(self, x):
        # x: [Batch, Channel, Height, Width]
        b, c, h, w = x.size()
        
        # 1. Polyphase Decomposition
        phases = []
        for i in range(self.stride):
            for j in range(self.stride):
                p = x[:, :, i::self.stride, j::self.stride]
                phases.append(p)

        # Handle edge cases if input dims aren't perfectly divisible
        if phases:
            min_h = min(p.size(2) for p in phases)
            min_w = min(p.size(3) for p in phases)
            phases = [p[:, :, :min_h, :min_w] for p in phases]

        # Stack: [Batch, 4, Channel, H', W']
        stack = torch.stack(phases, dim=1)
        
        # 2. Predict Weights based on global context
        weights = self.weight_gen(x) # [B, 4, 1, 1]
        weights = weights.view(b, self.stride * self.stride, 1, 1, 1)
        
        # 3. Weighted Sum (Learnable Downsampling)
        out = (stack * weights).sum(dim=1)
        
        return out

class TIPSTSMResNet18(nn.Module):
    """
    The Improved Model: TSM ResNet18 with TIPS injected into the stem.
    """
    def __init__(self, num_classes=27, n_segments=12):
        super().__init__()
        self.n_segments = n_segments
        
        # 1. Load Backbone
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. Inject TIPS into the ResNet Stem
        stem_channels = self.base_model.bn1.num_features # Typically 64
        self.base_model.maxpool = TIPS(in_channels=stem_channels, stride=2)
        
        # 3. Modify Classifier for Jester classes
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)
        
        # 4. Initialize TSM Module
        self.tsm = TSM(n_segment=n_segments, fold_div=8)

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)

        # --- Stem with TIPS ---
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x) # Calls TIPS

        # --- Layer 1 ---
        x = self.base_model.layer1(x)

        # --- Layer 2 (TSM) ---
        x = self.tsm(x)
        x = self.base_model.layer2(x)

        # --- Layer 3 (TSM) ---
        x = self.tsm(x)
        x = self.base_model.layer3(x)

        # --- Layer 4 (TSM) ---
        x = self.tsm(x)
        x = self.base_model.layer4(x)

        # --- Head ---
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)

        # --- Consensus ---
        x = x.view(b, s, -1)
        return x.mean(dim=1)