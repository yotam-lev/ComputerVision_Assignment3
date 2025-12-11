import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class JesterDataset(Dataset):
    def __init__(self, csv_file, root_dir, labels_file, transform=None, n_segments=1):
        # Force paths to be absolute to avoid relative path confusion
        self.root_dir = os.path.abspath(root_dir)
        
        # Read annotations
        self.annotations = pd.read_csv(csv_file, sep=';', header=None, names=['video_id', 'label'])
        self.transform = transform
        self.n_segments = n_segments
        self.label_map = self._create_label_map(labels_file)

    def _create_label_map(self, labels_file):
        df_labels = pd.read_csv(labels_file, header=None)
        return {label: idx for idx, label in enumerate(df_labels[0])}

    def _sample_indices(self, num_frames):
        indices = []
        if num_frames < self.n_segments:
            return [i % num_frames for i in range(self.n_segments)]
        segment_duration = num_frames // self.n_segments
        for i in range(self.n_segments):
            start = i * segment_duration
            # Center sampling
            indices.append(start + (segment_duration // 2))
        return indices

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # FIX: Ensure ID is int first to avoid "105128.0" strings
        video_id_raw = self.annotations.iloc[idx]['video_id']
        video_id = str(int(video_id_raw))
        label_str = self.annotations.iloc[idx]['label']
        
        # FIX: Construct absolute path
        video_path = os.path.join(self.root_dir, video_id)
        
        label_idx = self.label_map.get(label_str, -1)

        # DEBUG CHECK
        if not os.path.exists(video_path):
            print(f"❌ FAIL: Script expected folder at this EXACT path:\n   {video_path}")
            # Return dummy data so training doesn't crash immediately, but warns you
            return torch.zeros((self.n_segments, 3, 224, 224)), label_idx

        try:
            frames = sorted([f for f in os.listdir(video_path) if f.lower().endswith('.jpg')])
        except Exception as e:
            print(f"❌ Error listing files in {video_path}: {e}")
            return torch.zeros((self.n_segments, 3, 224, 224)), label_idx

        num_frames = len(frames)
        if num_frames == 0:
            print(f"⚠️ Warning: Folder {video_id} exists but contains no .jpg files.")
            return torch.zeros((self.n_segments, 3, 224, 224)), label_idx

        indices = self._sample_indices(num_frames)
        image_stack = []

        for i in indices:
            img_path = os.path.join(video_path, frames[i])
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                image_stack.append(img)
            except Exception as e:
                print(f"Error loading frame {img_path}: {e}")
                image_stack.append(torch.zeros(3, 224, 224))

        return torch.stack(image_stack), label_idx