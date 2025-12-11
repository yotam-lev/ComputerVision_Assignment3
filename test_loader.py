

from data.dataset import JesterDataset
from torchvision import transforms

# Define simple transform
tx = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Initialize Dataset (Make sure paths match your `datasets` folder structure)
ds = JesterDataset(
    csv_file='./datasets/jester-v1-small-train.csv', 
    root_dir='./small-20bn-jester-v1',
    labels_file='./datasets/jester-v1-labels.csv',
    transform=tx,
    n_segments=8  # Testing TSM mode
)

# Get one sample
frames, label = ds[0]
print(f"Output Shape: {frames.shape}") # Should be [8, 3, 224, 224]
print(f"Label Index: {label}")