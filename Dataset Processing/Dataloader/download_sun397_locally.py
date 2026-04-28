import os
import gdown
from pathlib import Path
from datasets import load_dataset

def main():
    # 1. Setup the CoOp folder structure
    base_dir = Path("sun397")
    sun_dir = base_dir / "SUN397"
    sun_dir.mkdir(parents=True, exist_ok=True)

    # 2. Download the official CoOp JSON split
    json_dest = base_dir / "split_zhou_SUN397.json"
    print(f"Downloading {json_dest.name}...")
    gdown.download(id="1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq", output=str(json_dest), quiet=False)

    # 3. Download and unpack images from HuggingFace
    print("\nDownloading SUN397 images from HuggingFace (~39GB unpacked)...")
    print("Depending on your internet speed, this may take a while.")
    ds = load_dataset("1aurent/SUN397", split="train", trust_remote_code=True)
    label_names = ds.features["label"].names

    per_class = {}
    for i, example in enumerate(ds):
        idx = example["label"]
        # HuggingFace labels have a leading slash (e.g., "/a/abbey"). Strip it.
        cat = label_names[idx].lstrip("/") 
        
        class_dir = sun_dir / cat
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep track of counts per class to recreate the exact naming convention
        per_class[idx] = per_class.get(idx, 0) + 1
        img_name = f"sun_{idx:03d}_{per_class[idx]:05d}.jpg"
        img_path = class_dir / img_name
        
        # Save image if it doesn't already exist
        if not img_path.exists():
            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(str(img_path), "JPEG", quality=95)
            
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,} / {len(ds):,} images...")

    print("\nSuccess! Your SUN397 dataset is properly structured.")

if __name__ == "__main__":
    main()