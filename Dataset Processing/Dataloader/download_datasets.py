#!/usr/bin/env python3
"""
Download script for CoOp benchmark datasets.
Follows the EXACT procedure from:
  https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md

Excluded (as per project spec):
  - ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R

Included datasets and their EXACT folder names under $DATA:
  caltech-101/        oxford_pets/        stanford_cars/
  oxford_flowers/     food-101/           fgvc_aircraft/
  sun397/             dtd/                eurosat/
  ucf101/             imagenet/  (optional — will prompt)

Usage:
    pip install tqdm gdown requests colorama
    python download_coop_datasets.py --root /path/to/DATA
    python download_coop_datasets.py --root /path/to/DATA --no-imagenet
    python download_coop_datasets.py --root /path/to/DATA --datasets dtd eurosat
"""

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen

# optional tqdm
try:
    from tqdm import tqdm
    def _reporthook(t):
        last = [0]
        def inner(count, block_size, total_size):
            if total_size not in (None, -1):
                t.total = total_size
            transferred = count * block_size
            t.update(transferred - last[0])
            last[0] = transferred
        return inner
except ImportError:
    tqdm = None
    def _reporthook(_):
        return None

# optional requests (better HEAD checks)
try:
    import requests as _req
    def _head_ok(url: str, timeout: int = 8) -> bool:
        """Try HEAD first; fall back to a GET range request.
        Many academic servers (Caltech, Stanford, ETH, Princeton) block HEAD
        but serve GET normally."""
        headers = {"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"}
        try:
            r = _req.head(url, timeout=timeout, allow_redirects=True, headers=headers)
            if r.status_code < 400:
                return True
        except Exception:
            pass
        # HEAD failed or blocked — try a tiny GET range request
        try:
            r = _req.get(url, timeout=timeout, allow_redirects=True,
                         headers=headers, stream=True)
            r.close()
            return r.status_code in (200, 206)
        except Exception:
            return False
except ImportError:
    from urllib.request import Request as _Req
    def _head_ok(url: str, timeout: int = 8) -> bool:
        """Try HEAD, then fall back to GET range."""
        for method in ("HEAD", "GET"):
            try:
                req = _Req(url, method=method,
                           headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"})
                conn = urlopen(req, timeout=timeout)
                conn.close()
                return True
            except Exception:
                continue
        return False

# colour helpers
try:
    import colorama
    colorama.init(autoreset=True)
    GREEN  = colorama.Fore.GREEN
    RED    = colorama.Fore.RED
    YELLOW = colorama.Fore.YELLOW
    CYAN   = colorama.Fore.CYAN
    BOLD   = colorama.Style.BRIGHT
    RESET  = colorama.Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""

DATASET_FILES = {

    # caltech-101
    # Expected:  caltech-101/101_ObjectCategories/
    #            caltech-101/split_zhou_Caltech101.json
    "caltech-101": [
        {
            "filename":        "caltech-101.zip",
            "mirrors":         [
                # Official CaltechDATA host — zip contains caltech-101/101_ObjectCategories/ at top level
                "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1",
            ],
            "gdrive_id":       None,
            "extract_to":      "__ROOT__",   # zip has top-level caltech-101/ → extract to $DATA
            "extracted_check": "__ROOT__:caltech-101/101_ObjectCategories",
        },
        {
            "filename":   "split_zhou_Caltech101.json",
            "mirrors":    [],
            "gdrive_id":  "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN",
            "extract_to": None,
        },
    ],

    # Expected:  oxford_pets/images/
    #            oxford_pets/annotations/
    #            oxford_pets/split_zhou_OxfordPets.json
    "oxford_pets": [
        {
            "filename":        "images.tar.gz",
            "mirrors":         ["https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"],
            "gdrive_id":       None,
            "extract_to":      None,
            "extracted_check": "images",        # oxford_pets/images/
        },
        {
            "filename":        "annotations.tar.gz",
            "mirrors":         ["https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"],
            "gdrive_id":       None,
            "extract_to":      None,
            "extracted_check": "annotations",   # oxford_pets/annotations/
        },
        {
            "filename":   "split_zhou_OxfordPets.json",
            "mirrors":    [],
            "gdrive_id":  "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs",
            "extract_to": None,
        },
    ],

    # Expected:  stanford_cars/cars_train/
    #            stanford_cars/cars_test/
    #            stanford_cars/devkit/
    #            stanford_cars/cars_test_annos_withlabels.mat
    #            stanford_cars/split_zhou_StanfordCars.json
    #
    # NOTE: The original Stanford AI Lab URLs are permanently offline (confirmed
    # by PyTorch/torchvision docs). Images are downloaded via the Kaggle CLI:
    #   pip install kaggle
    #   Place ~/.kaggle/kaggle.json (from kaggle.com → Account → API token)
    # Kaggle dataset: rickyyyyyyy/torchvision-stanford-cars
    #   TorchVision-compatible — extracts directly into the expected structure.
    "stanford_cars": [
        {
            "filename":        "__kaggle__",    # sentinel — handled by kaggle_download()
            "mirrors":         [],
            "gdrive_id":       None,
            "extract_to":      None,
            "extracted_check": "cars_train",
            "kaggle_dataset":  "rickyyyyyyy/torchvision-stanford-cars",
        },
        {
            "filename":        "split_zhou_StanfordCars.json",
            "mirrors":         [],
            "gdrive_id":       "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT",
            "extract_to":      None,
            "extracted_check": None,
        },
    ],

    # Expected:  oxford_flowers/jpg/
    #            oxford_flowers/imagelabels.mat
    #            oxford_flowers/cat_to_name.json
    #            oxford_flowers/split_zhou_OxfordFlowers.json
    "oxford_flowers": [
        {
            "filename":        "102flowers.tgz",
            "mirrors":         ["https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"],
            "gdrive_id":       None,
            "extract_to":      None,
            "extracted_check": "jpg",           # oxford_flowers/jpg/
        },
        {
            "filename":   "imagelabels.mat",
            "mirrors":    ["https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"],
            "gdrive_id":  None,
            "extract_to": None,
        },
        {
            "filename":   "cat_to_name.json",
            "mirrors":    [],
            "gdrive_id":  "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0",
            "extract_to": None,
        },
        {
            "filename":   "split_zhou_OxfordFlowers.json",
            "mirrors":    [],
            "gdrive_id":  "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT",
            "extract_to": None,
        },
    ],

    # Expected:  food-101/images/
    #            food-101/meta/
    #            food-101/split_zhou_Food101.json
    # NOTE: the archive itself contains a top-level food-101/ folder, so we
    #       extract to $DATA (the root) to avoid food-101/food-101/ nesting.
    "food-101": [
        {
            "filename":        "food-101.tar.gz",
            "mirrors":         [
                "https://huggingface.co/datasets/ethz/food101/resolve/main/food-101.tar.gz",
                "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/food-101.tar.gz",
            ],
            "gdrive_id":       None,
            "extract_to":      "__ROOT__",
            "extracted_check": "__ROOT__:food-101/images",  # $DATA/food-101/images/
        },
        {
            "filename":   "split_zhou_Food101.json",
            "mirrors":    [],
            "gdrive_id":  "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl",
            "extract_to": None,
        },
    ],

    # Per spec: extract, keep only data/, rename to fgvc_aircraft/
    # Expected:  fgvc_aircraft/images/
    #            fgvc_aircraft/*.txt
    # NOTE: archive contains fgvc-aircraft-2013b/data/ — we post-process.
    "fgvc_aircraft": [
        {
            "filename":        "fgvc-aircraft-2013b.tar.gz",
            "mirrors":         [
                "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
            ],
            "gdrive_id":       None,
            "extract_to":      "__SCRATCH__",
            "extracted_check": "images",        # fgvc_aircraft/images/
        },

    ],

    # Expected:  sun397/SUN397/
    #            sun397/split_zhou_SUN397.json
    #
    # NOTE: Original mirrors are permanently offline:
    #   http://groups.csail.mit.edu/vision/SUN1old/SUN397.tar
    #   http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
    # Images are downloaded via HuggingFace datasets library:
    #   pip install datasets
    #   Dataset: 1aurent/SUN397 (108,754 images, all 397 categories)
    # Partitions.zip (also dead) is not needed — CoOp uses split_zhou_SUN397.json.
    "sun397": [
        {
            "filename":        "__huggingface__",   # sentinel — handled by huggingface_download_sun397()
            "mirrors":         [],
            "gdrive_id":       None,
            "extract_to":      None,
            "extracted_check": "SUN397",             # sun397/SUN397/
            "hf_dataset":      "1aurent/SUN397",
        },
        {
            "filename":   "split_zhou_SUN397.json",
            "mirrors":    [],
            "gdrive_id":  "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq",
            "extract_to": None,
        },
    ],

    # Expected:  dtd/images/
    #            dtd/imdb/
    #            dtd/labels/
    #            dtd/split_zhou_DescribableTextures.json
    # NOTE: archive contains a top-level dtd/ folder → extract to $DATA root.
    "dtd": [
        {
            "filename":        "dtd-r1.0.1.tar.gz",
            "mirrors":         ["https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"],
            "gdrive_id":       None,
            "extract_to":      "__ROOT__",
            "extracted_check": "__ROOT__:dtd/images",   # $DATA/dtd/images/
        },
        {
            "filename":   "split_zhou_DescribableTextures.json",
            "mirrors":    [],
            "gdrive_id":  "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x",
            "extract_to": None,
        },
    ],

    # Expected:  eurosat/2750/
    #            eurosat/split_zhou_EuroSAT.json
    "eurosat": [
        {
            "filename":        "EuroSAT.zip",
            "mirrors":         [
                "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip",
                "http://madm.dfki.de/files/sentinel/EuroSAT.zip",
            ],
            "gdrive_id":       None,
            "extract_to":      None,
            "extracted_check": "2750",          # eurosat/2750/
        },
        {
            "filename":   "split_zhou_EuroSAT.json",
            "mirrors":    [],
            "gdrive_id":  "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o",
            "extract_to": None,
        },
    ],

    # ucf101
    # Expected:  ucf101/UCF-101-midframes/
    #            ucf101/split_zhou_UCF101.json
    "ucf101": [
        {
            "filename":        "UCF-101-midframes.zip",
            "mirrors":         [],
            "gdrive_id":       "10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O",
            "extract_to":      None,
            "extracted_check": "UCF-101-midframes",  # ucf101/UCF-101-midframes/
        },
        {
            "filename":   "split_zhou_UCF101.json",
            "mirrors":    [],
            "gdrive_id":  "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y",
            "extract_to": None,
        },
    ],
}


IMAGENET_NOTE = "ImageNet — Manual Download Required"

# -----------------------------------------------------------------------------
# Phase 1 — URL verification
# -----------------------------------------------------------------------------

def check_all_urls(selected: dict) -> dict:
    print(f"\n{BOLD}{CYAN}{'='*64}")
    print(f"  Phase 1 — Verifying all download URLs …")
    print(f"{'='*64}{RESET}\n")

    report = {}
    for ds_name, files in selected.items():
        print(f"  {BOLD}{ds_name}{RESET}")
        report[ds_name] = []

        for f in files:
            fname     = f["filename"]
            mirrors   = f["mirrors"]
            gdrive_id = f["gdrive_id"]
            entry = {"filename": fname, "status": None, "url": None,
                     "gdrive_id": gdrive_id, "extract_to": f.get("extract_to")}

            if fname == "__kaggle__":
                kg_ds = f.get("kaggle_dataset", "")
                entry["status"]     = "kaggle"
                entry["url"]        = f"https://www.kaggle.com/datasets/{kg_ds}"
                entry["kaggle_dataset"] = kg_ds
                entry["extracted_check"] = f.get("extracted_check")
                print(f"    {YELLOW}[kaggle ]{RESET}  {kg_ds}")
            elif fname == "__huggingface__":
                hf_ds = f.get("hf_dataset", "")
                entry["status"]     = "huggingface"
                entry["url"]        = f"https://huggingface.co/datasets/{hf_ds}"
                entry["hf_dataset"]  = hf_ds
                entry["extracted_check"] = f.get("extracted_check")
                print(f"    {YELLOW}[hf     ]{RESET}  {hf_ds}")
            elif gdrive_id:
                entry["status"] = "gdrive"
                entry["url"]    = f"https://drive.google.com/uc?id={gdrive_id}"
                print(f"    {YELLOW}[gdrive ]{RESET}  {fname}")
            else:
                resolved = False
                for url in mirrors:
                    short = (url[:55] + "…") if len(url) > 55 else url
                    sys.stdout.write(f"    [checking]  {fname:<48}  {short}")
                    sys.stdout.flush()
                    ok = _head_ok(url)
                    if ok:
                        entry["status"] = "ok"
                        entry["url"]    = url
                        print(f"\r    {GREEN}[  ok   ]{RESET}  {fname:<48}  {short}")
                        resolved = True
                        break
                    else:
                        print(f"\r    {YELLOW}[ retry ]{RESET}  {fname:<48}  failed: {short}")

                if not resolved:
                    entry["status"] = "unavailable"
                    print(f"    {RED}[ FAIL  ]{RESET}  {fname} — no working mirror found!")

            report[ds_name].append(entry)
        print()

    return report


def print_url_summary(report: dict) -> bool:
    print(f"\n{BOLD}{CYAN}{'='*64}")
    print(f"  URL Check Summary")
    print(f"{'='*64}{RESET}\n")
    print(f"  {'Dataset':<20}  {'Direct':>7}  {'GDrive':>7}  {'Kaggle':>6}  {'HF':>4}  {'Missing':>8}")
    print(f"  {'─'*20}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*4}  {'─'*8}")

    all_ok = True
    for ds_name, entries in report.items():
        n_ok     = sum(1 for e in entries if e["status"] == "ok")
        n_gdrive = sum(1 for e in entries if e["status"] == "gdrive")
        n_kg     = sum(1 for e in entries if e["status"] == "kaggle")
        n_hf     = sum(1 for e in entries if e["status"] == "huggingface")
        unavail  = [e["filename"] for e in entries if e["status"] == "unavailable"]
        icon     = f"{GREEN}✓{RESET}" if not unavail else f"{RED}✗{RESET}"
        miss     = f"{RED}{len(unavail)}{RESET}" if unavail else f"{GREEN}0{RESET}"
        print(f"  {icon} {ds_name:<20}  {n_ok:>7}  {n_gdrive:>7}  {n_kg:>6}  {n_hf:>4}  {miss:>8}")
        for fn in unavail:
            print(f"      {RED}↳ unavailable: {fn}{RESET}")
            all_ok = False

    print()
    if all_ok:
        print(f"  {GREEN}{BOLD}All files found — ready to download.{RESET}\n")
    else:
        print(f"  {YELLOW}{BOLD}Some files unavailable — they will be skipped.{RESET}\n")
    return all_ok


# -----------------------------------------------------------------------------
# Phase 2 — Downloading & extraction helpers
# -----------------------------------------------------------------------------
def _already_present(fp: Path, extracted_check: str, ds_root: Path, data_root: Path) -> bool:
    """
    Two-step existence check:
      1. Is the archive file itself on disk?
      2. If not, does the extracted output already exist?

    extracted_check syntax:
      "some/rel/path"          → relative to ds_root
      "__ROOT__:some/rel/path" → relative to data_root
    """
    # Step 1 — archive present?
    if fp.exists():
        return True

    # Step 2 — extracted output present?
    if not extracted_check:
        return False

    if extracted_check.startswith("__ROOT__:"):
        rel = extracted_check[len("__ROOT__:"):]
        check_path = data_root / rel
    else:
        check_path = ds_root / extracted_check

    if check_path.exists():
        print(f"  {YELLOW}[skip    ]{RESET}  {fp.name} — extracted output already found at {check_path.relative_to(data_root)}")
        return True

    return False


def download_file(url: str, dest: Path, label: str = "",
                  extracted_check: str = "", ds_root: Path = None, data_root: Path = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _already_present(dest, extracted_check, ds_root or dest.parent, data_root or dest.parent):
        if dest.exists():
            print(f"  {YELLOW}[skip    ]{RESET}  {dest.name} archive already on disk.")
        return
    tag = label or dest.name
    if tqdm:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
                  miniters=1, desc=f"    {tag[:50]}") as t:
            urlretrieve(url, dest, reporthook=_reporthook(t))
    else:
        print(f"  {CYAN}[download]{RESET}  {tag}")
        urlretrieve(url, dest)


def gdrive_download(gdrive_id: str, dest: Path, label: str = "",
                    extracted_check: str = "", ds_root: Path = None, data_root: Path = None):
    if _already_present(dest, extracted_check, ds_root or dest.parent, data_root or dest.parent):
        if dest.exists():
            print(f"  {YELLOW}[skip    ]{RESET}  {dest.name} archive already on disk.")
        return
    try:
        import gdown
    except ImportError:
        print(
            f"  {RED}[no gdown]{RESET}  Cannot download {dest.name}.\n"
            f"    → pip install gdown  then re-run, or download manually:\n"
            f"      https://drive.google.com/uc?id={gdrive_id}\n"
        )
        return
    tag = label or dest.name
    print(f"  {CYAN}[gdown   ]{RESET}  {tag}")
    gdown.download(id=gdrive_id, output=str(dest), quiet=False)


def extract_archive(archive: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {CYAN}[extract ]{RESET}  {archive.name}  →  {dest_dir}")
    name = archive.name
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive type: {name}")



def kaggle_download(dataset: str, dest_dir: Path, extracted_check: str = "",
                    ds_root: Path = None, data_root: Path = None):
    """Download a Kaggle dataset using the kaggle CLI."""
    if extracted_check and _already_present(
        dest_dir / "__kaggle__", extracted_check,
        ds_root or dest_dir, data_root or dest_dir
    ):
        return

    try:
        import kaggle  # noqa: F401 — triggers credential check
    except ImportError:
        print(
            "  [no kaggle]  kaggle package not found.\n"
            "    pip install kaggle\n"
            "    Get API token: kaggle.com -> Account -> Create API Token\n"
            "    Place at: ~/.kaggle/kaggle.json"
        )
        return
    except Exception as e:
        print(f"  [kaggle error]  {e}")
        print("    Ensure ~/.kaggle/kaggle.json exists with valid credentials.")
        return

    import subprocess
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {CYAN}[kaggle  ]{RESET}  Downloading {dataset} …")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "--unzip", "-p", str(dest_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  {RED}[kaggle fail]{RESET}  {result.stderr.strip()}")
    else:
        print(f"  {GREEN}[kaggle  ]{RESET}  Downloaded {dataset}")


def huggingface_download_sun397(ds_root: Path, data_root: Path):
    """
    Download SUN397 images from HuggingFace (1aurent/SUN397) and reconstruct
    the expected folder structure:  sun397/SUN397/<category_path>/sun_<idx>.jpg

    The original Princeton/MIT mirrors are permanently dead. This function
    uses the HuggingFace `datasets` library as a drop-in replacement.
    """
    sun_dir = ds_root / "SUN397"
    if sun_dir.exists() and any(sun_dir.rglob("sun_*.jpg")):
        print(f"  {YELLOW}[skip    ]{RESET}  SUN397/ already populated.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            f"  {RED}[no datasets]{RESET}  Cannot download SUN397 via HuggingFace.\n"
            f"    → pip install datasets  then re-run, or download manually:\n"
            f"      https://huggingface.co/datasets/1aurent/SUN397\n"
        )
        return

    print(f"  {CYAN}[hf     ]{RESET}  Downloading 1aurent/SUN397 … (108,754 images, this may take a while)")
    ds = load_dataset("1aurent/SUN397", split="train", trust_remote_code=True)

    # Get label names — they look like "/a/abbey", "/a/airplane_cabin", etc.
    label_names = ds.features["label"].names

    saved = 0
    per_class_count = {}
    for i, example in enumerate(ds):
        label_idx = example["label"]
        label_name = label_names[label_idx]          # e.g. "/a/abbey"
        # Strip leading slash for path construction
        category_path = label_name.lstrip("/")       # e.g. "a/abbey"

        class_dir = sun_dir / category_path
        class_dir.mkdir(parents=True, exist_ok=True)

        # Generate sun_XXXXXX.jpg filename with per-class counter
        per_class_count[label_idx] = per_class_count.get(label_idx, 0) + 1
        img_name = f"sun_{label_idx:03d}_{per_class_count[label_idx]:05d}.jpg"

        img_path = class_dir / img_name
        if not img_path.exists():
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(str(img_path), "JPEG", quality=95)

        saved += 1
        if saved % 5000 == 0:
            print(f"           {saved:,} / 108,754 images saved …")

    print(f"  {GREEN}[hf     ]{RESET}  Saved {saved:,} images to {sun_dir}")


def post_process_stanford_cars(ds_root: Path):
    """
    Post-process Stanford Cars after Kaggle download.
    The rickyyyyyyy/torchvision-stanford-cars dataset is TorchVision-compatible
    and should extract directly into the expected structure (cars_train/, cars_test/,
    devkit/, etc.). This function verifies the structure is correct.
    """
    for expected in ("cars_train", "cars_test"):
        p = ds_root / expected
        if p.exists():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {GREEN}[verify  ]{RESET}  {expected}/ — {n:,} files")
        else:
            print(f"  {YELLOW}[warn    ]{RESET}  {expected}/ not found — check Kaggle download")

def post_process_fgvc(ds_root: Path, data_root: Path):
    """
    Per DATASETS.md:
      Extract fgvc-aircraft-2013b.tar.gz, keep only data/, rename to fgvc_aircraft/.
    The archive extracts to a temp dir; we move data/ to fgvc_aircraft/ under $DATA.
    """
    scratch = ds_root.parent / "_fgvc_scratch"
    archive = ds_root / "fgvc-aircraft-2013b.tar.gz"
    if not archive.exists():
        return

    if not (ds_root / "images").exists():
        print(f"  {CYAN}[post   ]{RESET}  Restructuring fgvc_aircraft …")
        scratch.mkdir(parents=True, exist_ok=True)
        extract_archive(archive, scratch)
        data_src = scratch / "fgvc-aircraft-2013b" / "data"
        if data_src.exists():
            for item in data_src.iterdir():
                dest = ds_root / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
        shutil.rmtree(scratch, ignore_errors=True)


# -----------------------------------------------------------------------------
# Per-dataset download orchestrator
# -----------------------------------------------------------------------------
def download_dataset(ds_name: str, data_root: Path, report: dict):
    ds_root = data_root / ds_name
    ds_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}{'─'*64}{RESET}")
    print(f"  {BOLD}{ds_name}{RESET}  →  {ds_root}")
    print(f"{'─'*64}")

    for entry in report[ds_name]:
        fname      = entry["filename"]
        status     = entry["status"]
        url        = entry["url"]
        gdid       = entry["gdrive_id"]
        extract_to = entry.get("extract_to")
        fp         = ds_root / fname

        if status == "unavailable":
            print(f"  {RED}[skip    ]{RESET}  {fname} — no working mirror.")
            continue

        ec = entry.get("extracted_check") or ""

        # Download
        if fname == "__huggingface__":
            huggingface_download_sun397(ds_root, data_root)
            continue  # images saved directly, no archive to extract
        elif fname == "__kaggle__":
            kaggle_download(
                entry.get("kaggle_dataset", ""),
                ds_root,
                extracted_check=ec, ds_root=ds_root, data_root=data_root,
            )
            continue  # no archive to extract
        elif status == "gdrive":
            gdrive_download(gdid, fp, fname,
                            extracted_check=ec, ds_root=ds_root, data_root=data_root)
        else:
            download_file(url, fp, fname,
                          extracted_check=ec, ds_root=ds_root, data_root=data_root)

        # Extract — only if archive was actually downloaded (and not already extracted)
        if fp.exists() and any(fname.endswith(ext) for ext in (".tar.gz", ".tgz", ".tar", ".zip")):
            # Don't re-extract if the output folder already exists
            if ec and not ec.startswith("__ROOT__:"):
                already_extracted = (ds_root / ec).exists()
            elif ec.startswith("__ROOT__:"):
                already_extracted = (data_root / ec[len("__ROOT__:"):]).exists()
            else:
                already_extracted = False

            if already_extracted:
                print(f"  {YELLOW}[skip    ]{RESET}  {fname} — already extracted.")
            elif extract_to == "__ROOT__":
                extract_archive(fp, data_root)
            elif extract_to == "__SCRATCH__":
                pass  # handled by post_process below
            else:
                extract_archive(fp, ds_root)

    # Post-processing for datasets that need restructuring
    if ds_name == "fgvc_aircraft":
        post_process_fgvc(ds_root, data_root)
    if ds_name == "stanford_cars":
        post_process_stanford_cars(ds_root)

    print(f"  {GREEN}✓  {ds_name} complete.{RESET}")


# -----------------------------------------------------------------------------
# ImageNet prompt
# -----------------------------------------------------------------------------
def ask_imagenet(data_root: Path, force_yes: bool = False, force_no: bool = False) -> bool:
    print(f"\n{BOLD}{CYAN}{'='*64}")
    print(f"  ImageNet — Optional (~144 GB, manual registration required)")
    print(f"{'='*64}{RESET}")
    print(
        f"\n  {RED}{BOLD}WARNING:{RESET} ImageNet is ~144 GB and requires manual registration.\n"
        f"  It will {RED}NOT fit on Google Colab{RESET} (~100 GB disk limit).\n"
        f"  All prompting experiments can be run without it.\n"
    )

    if force_no:
        print(f"  {YELLOW}Skipping ImageNet (--no-imagenet).{RESET}")
        return False
    if force_yes:
        print(IMAGENET_NOTE)
        (data_root / "imagenet" / "images").mkdir(parents=True, exist_ok=True)
        return True

    while True:
        ans = input(f"  {BOLD}Include ImageNet? [y/N]: {RESET}").strip().lower()
        if ans in ("", "n", "no"):
            print(f"  {YELLOW}Skipping ImageNet.{RESET}")
            return False
        if ans in ("y", "yes"):
            print(IMAGENET_NOTE)
            (data_root / "imagenet" / "images").mkdir(parents=True, exist_ok=True)
            print(f"  {GREEN}Created: {data_root / 'imagenet' / 'images'}{RESET}")
            print(f"  Follow the instructions above to complete the download.")
            return True
        print("  Please enter y or n.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CoOp datasets following DATASETS.md exactly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset folder names (as created under $DATA):
  caltech-101   oxford_pets   stanford_cars   oxford_flowers
  food-101      fgvc_aircraft sun397          dtd
  eurosat       ucf101

Examples:
  python download_coop_datasets.py --root /data
  python download_coop_datasets.py --root /data --no-imagenet
  python download_coop_datasets.py --root /data --datasets dtd eurosat ucf101
        """,
    )
    parser.add_argument("--root", required=False, default=None,
                        help="$DATA root directory. If omitted, you will be prompted interactively.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_FILES.keys()),
                        default=list(DATASET_FILES.keys()), metavar="DS",
                        help="Specific datasets to download (default: all).")
    parser.add_argument("--yes-imagenet", action="store_true",
                        help="Include ImageNet without prompting.")
    parser.add_argument("--no-imagenet",  action="store_true",
                        help="Skip ImageNet without prompting.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve root — default to a 'coop_data' folder next to this script
    if args.root is None:
        data_root = Path(__file__).parent.resolve() / "coop_data"
    else:
        data_root = Path(args.root).expanduser().resolve()

    data_root.mkdir(parents=True, exist_ok=True)

    selected = {k: DATASET_FILES[k] for k in args.datasets}

    print(f"\n{BOLD}{CYAN}{'='*64}")
    print(f"  CoOp Dataset Downloader  (follows DATASETS.md exactly)")
    print(f"{'='*64}{RESET}")
    print(f"  $DATA    : {data_root}")
    print(f"  Datasets : {', '.join(selected)}")

    # Verify URLs
    report = check_all_urls(selected)
    all_ok = print_url_summary(report)

    # ImageNet
    include_imagenet = ask_imagenet(
        data_root,
        force_yes=args.yes_imagenet,
        force_no=args.no_imagenet,
    )

    # Confirm
    print(f"\n{BOLD}{CYAN}{'='*64}")
    print(f"  Phase 2 — Download")
    print(f"{'='*64}{RESET}\n")
    confirm = input(f"  {BOLD}Proceed with download? [Y/n]: {RESET}").strip().lower()
    if confirm in ("n", "no"):
        print("  Aborted.")
        return

    # Download
    failed = []
    for ds_name in selected:
        try:
            download_dataset(ds_name, data_root, report)
        except Exception as exc:
            print(f"  {RED}✗  {ds_name} FAILED: {exc}{RESET}")
            failed.append(ds_name)

    # Final summary
    succeeded = [n for n in selected if n not in failed]
    print(f"\n{BOLD}{CYAN}{'='*64}")
    print(f"  All Done")
    print(f"{'='*64}{RESET}")
    print(f"  {GREEN}Succeeded ({len(succeeded)}): {', '.join(succeeded) or '—'}{RESET}")
    if failed:
        print(f"  {RED}Failed    ({len(failed)}): {', '.join(failed)}{RESET}")
    if include_imagenet:
        print(f"\n  {YELLOW}⚠  Complete ImageNet download manually — see instructions above.{RESET}")
    if not all_ok:
        print(f"\n  {YELLOW}⚠  Some files had no working mirror — check output above.{RESET}")
    print(
        f"\n  {BOLD}Tip:{RESET} Install gdown for automatic Google Drive downloads:\n"
        f"    pip install gdown\n"
    )

if __name__ == "__main__":
    main()