import os
import torch
from PIL import Image

class FGVCAircraftDataset(torch.utils.data.Dataset):
    """FGVC Aircraft - flat images/ dir, official variant txt splits on Drive.
    Each line format: '{image_id} {class_name}'  (class_name may contain spaces).
    options: 'fgvc_aircraft'
    """
    def __init__(self, img_dir, samples, transform=None):
        self.img_dir, self.samples, self.transform = img_dir, samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        return self.transform(img) if self.transform else img, label

class GenericJSONSplitDataset(torch.utils.data.Dataset):
    """Generic dataset loader that uses a JSON file for splits.
    Handles flat or nested directories depending on fname in the split.
    options: 'eurosat', 'caltech101', 'oxford_pets', 'dtd', 'flowers102', 'food101', 'stanford_cars', 'ucf101'
    """
    def __init__(self, img_dir, samples, transform=None):
        self.img_dir = img_dir
        self.samples = samples
        self.transform = transform
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        # Attempt to open directly; catch missing file to use fallback
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            fallback_path = os.path.join(self.img_dir, os.path.basename(fname))
            img = Image.open(fallback_path).convert('RGB')
            
        if self.transform:
            img = self.transform(img)
        return img, label
