import torch
import torch.nn as nn

class TSM(nn.Module):
    def __init__(self, n_segment=8, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        # x input: [Batch * Segments, Channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        
        # Reshape to [Batch, Segments, C, H, W] to access 'time' dimension
        x = x.view(n_batch, self.n_segment, c, h, w)
        
        # Calculate how many channels to shift (1/8th usually)
        fold = c // self.fold_div
        
        out = torch.zeros_like(x)
        
        # Shift Left (Past -> Current)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        
        # Shift Right (Future -> Current)
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        
        # No Shift (Center)
        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        
        # Flatten back to [Batch * Segments, C, H, W]
        return out.view(nt, c, h, w)