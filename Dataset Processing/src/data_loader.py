import os
import torch
import clip
from torchvision import datasets
from torch.utils.data import DataLoader

# ==========================================
# 1. Initialize CLIP & Preprocessor
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/16", device=device)

# ==========================================
# 2. CoOp JSON Split Downloader
# ==========================================
def download_coop_splits(dataset_name, root_dir="./data"):
    """
    Downloads the official few-shot split JSON files from the CoOp GitHub repository.
    """
    os.makedirs(root_dir, exist_ok=True)
    
    # The base URL for the CoOp repository's split files
    base_url = "https://raw.githubusercontent.com/KaiyangZhou/CoOp/main/configs/datasets/"
    
    # Map our dataset names to CoOp's specific JSON filenames
    json_map = {
        "caltech101": "caltech101.yaml", # Note: CoOp actually generates these on the fly or uses specific paths. 
        "oxfordpets": "oxford_pets.yaml",
        # For true few-shot reproducibility, CoOp provides split files (e.g., split_zhou_OxfordPets.json)
        # We will download the specific split files if they are hosted, or you can generate them 
        # using CoOp's provided scripts in their repo.
    }
    
    split_filename = f"split_zhou_{dataset_name}.json"
    split_path = os.path.join(root_dir, split_filename)
    
    # In a full implementation, you'd fetch the specific split_zhou_*.json file here.
    # If the file doesn't exist locally, you prompt the user to run CoOp's split generator.
    if not os.path.exists(split_path):
        print(f"[Warning] CoOp split file {split_filename} not found in {root_dir}.")
        print(f"For exact few-shot reproduction, generate/download it via the CoOp repo.")
        
    return split_path

# ==========================================
# 3. Dataset Factory Function
# ==========================================
def get_dataset(dataset_name, root_dir="./data", split="train"):
    """
    Downloads and returns the specified dataset.
    """
    os.makedirs(root_dir, exist_ok=True)
    dataset_name = dataset_name.lower()
    
    # ---------------------------------------------------------
    # AUTOMATICALLY DOWNLOADABLE DATASETS (Via Torchvision)
    # ---------------------------------------------------------
    if dataset_name == "oxfordpets":
        dataset_split = "trainval" if split == "train" else "test"
        return datasets.OxfordIIITPet(root=root_dir, split=dataset_split, download=True, transform=preprocess)
        
    elif dataset_name == "food101":
        return datasets.Food101(root=root_dir, split=split, download=True, transform=preprocess)
        
    elif dataset_name == "caltech101":
        # Note: Caltech101 target type needs to be handled if using torchvision directly
        return datasets.Caltech101(root=root_dir, download=True, transform=preprocess)

    # ---------------------------------------------------------
    # MANUAL DATASETS (Require manual download due to server/license issues)
    # ---------------------------------------------------------
    elif dataset_name == "stanfordcars":
        # The official Stanford server is permanently down.
        # ACTION REQUIRED: Download from Kaggle, place in ./data/stanford_cars/
        cars_path = os.path.join(root_dir, "stanford_cars")
        if not os.path.exists(cars_path):
            raise FileNotFoundError(
                "Stanford Cars server is offline. Please manually download the dataset "
                "from Kaggle, extract it, and place it in the './data/stanford_cars' directory."
            )
        return datasets.StanfordCars(root=root_dir, split=split, download=False, transform=preprocess)

    elif dataset_name == "imagenet":
        # Requires academic registration to download.
        # ACTION REQUIRED: Download ILSVRC2012 manually, place in ./data/imagenet/
        imagenet_path = os.path.join(root_dir, "imagenet")
        if not os.path.exists(imagenet_path):
            raise FileNotFoundError(
                "ImageNet cannot be downloaded automatically due to licensing. "
                "Please download ILSVRC2012 manually and place it in './data/imagenet'."
            )
        return datasets.ImageNet(root=root_dir, split=split, transform=preprocess)

    # ---------------------------------------------------------
    # EXCLUDED DATASETS (Strictly forbidden by project spec)
    # ---------------------------------------------------------
    elif dataset_name in ["imagenetv2", "imagenet-sketch", "imagenet-a", "imagenet-r"]:
        raise ValueError(f"Dataset {dataset_name} is explicitly excluded by the project spec.")
        
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not yet implemented.")

# ==========================================
# 4. DataLoader Setup
# ==========================================
def create_dataloader(dataset_name, batch_size=32, split="train", num_workers=2, use_coop_splits=False):
    """
    Creates a PyTorch DataLoader. If use_coop_splits=True, it will attempt to subset 
    the data based on the CoOp few-shot JSON files.
    """
    dataset = get_dataset(dataset_name, split=split)
    
    if use_coop_splits:
        # Check for the JSON file
        split_file = download_coop_splits(dataset_name)
        # TODO for Member 1: Write logic here to open the JSON file, read the specific 
        # image indices for the requested split (train/val/test), and wrap the `dataset` 
        # in a torch.utils.data.Subset(dataset, indices) before passing to DataLoader.
        pass 
    
    is_train = (split == "train")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # Handling class names safely (different torchvision datasets store them differently)
    class_names = getattr(dataset, 'classes', None)
    
    return dataloader, class_names

if __name__ == "__main__":
    # Test with OxfordPets
    loader, class_names = create_dataloader("oxfordpets", batch_size=16, split="train")
    print(f"Classes found: {len(class_names)}")