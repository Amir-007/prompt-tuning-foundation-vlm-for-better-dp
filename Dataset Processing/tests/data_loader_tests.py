import sys
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Add the project's `src` directory to the path so we can import the package modules.
# This makes the test runnable from the repo root without modifying the environment.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Import the DataLoader module from the package path under `src`.
from data_loader import create_dataloader

def test_oxford_pets_loading():
    print("Testing OxfordPets DataLoader...")
    loader, class_names = create_dataloader("oxfordpets", batch_size=4, split="train")
    images, labels = next(iter(loader))

    print(f"Images tensor shape: {images.shape}")
    print(f"Labels tensor shape: {labels.shape}")
    
    # Visualisation logic
    def imshow_clip(tensor, title=None):
        image = tensor.numpy().transpose((1, 2, 0))
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.axis('off')

    out = torchvision.utils.make_grid(images)
    class_labels = [class_names[label.item()] for label in labels]

    plt.figure(figsize=(10, 5))
    imshow_clip(out, title=" | ".join(class_labels))
    plt.show()

if __name__ == "__main__":
    test_oxford_pets_loading()