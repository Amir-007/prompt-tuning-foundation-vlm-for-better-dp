# Prompt Tuning CLIP: CoCoOp vs ProMIM

This repository contains the implementation for comparing two conditional prompt learning architectures applied to Vision-Language Models (specifically OpenAI's CLIP). The project explores the performance of **CoCoOp** (Conditional Context Optimization) versus **ProMIM** (Masked Image Modeling for Conditional Prompt Learning) across multiple few-shot learning settings.


## Overview

The main codebase is fully encapsulated within the Jupyter Notebook `EEEM068_CoCoOp_vs_ProMIM.ipynb`. Unlike many existing implementations that heavily rely on external vision-language training frameworks (like Dassl or the original CoOp repository), this notebook implements the training loop, context optimization layers, and prompt tuning logic from scratch using standard PyTorch and the official OpenAI CLIP package.


### Key Features
- **Standalone Implementations**: Complete structural implementations of CoCoOp and ProMIM, utilizing precise `"a photo of a"` context alignment for optimized Base-To-New structural generalisation.
- **Unified Configuration System**: A global `DATASET_CONFIGS` dictionary seamlessly handles configurations, splits, and folder mapping for 9 different classification benchmarks (EuroSAT, Caltech101, Oxford Pets, DTD, Flowers102, Food101, FGVC Aircraft, Stanford Cars, UCF101) directly inside the single master notebook.
- **Robust Dataset Handling**: Automated, JSON-based dataset loaders support the multiple classification benchmarks. Extracting archives and defining exact splits (Train/Val/Test) is handled dynamically based on the chosen `DATASET_NAME`.
- **Few-Shot Paradigm**: Automated sampling utilities to uniformly construct 1-shot, 4-shot, 8-shot, and 16-shot evaluation splits for few-shot performance analysis.
- **Optimized JSON Charting Engine**: Heavy evaluation operations are entirely decoupled from plotting routines. Generating performance line/bar charts now structurally maps pre-computed Harmonic Mean (`H`) Base-to-New metrics natively from standard `.json` logging—minimising memory constraints.
- **Extensive Visualizations**:
  - Comparative Text-to-Image **Attention Maps** (comparing Zero-Shot CLIP, CoCoOp, and ProMIM).
  - Multi-method comparison charts mapping few-shot performance trajectories.
  - **Embedding Space Visualizations** using PCA, t-SNE, and UMAP to inspect learned text/image representations.
- **Ablation Studies**: Built-in routines to test the effect of different mask ratios and gradient scaling hyperparameters ($\lambda$).


## Project Structure & Setup

```text
ProMIM/
├── code/
│   ├── EEEM068_CoCoOp_vs_ProMIM.ipynb  # Primary experimentation notebook
│   └── README.md                       # This file
├── Datasets/                           # Compressed datasets and JSON data splits
└── Extracted_Datasets/                 # Automatically generated extracted target directory
```


### Dependencies

To run the notebook, you will need a standard PyTorch environment and the following non-standard dependencies:

- **OpenAI CLIP**: `pip install git+https://github.com/openai/CLIP.git`
- **UMAP Learn** (for embedding visualizations): `pip install umap-learn`

Other standard libraries expected: `torch`, `torchvision`, `numpy`, `matplotlib`, `tqdm`. A CUDA-capable GPU is highly recommended.


## Running the Notebook

The notebook is divided into 13 logical sections. You can run them sequentially:

1. **Install dependencies & Setup Device**: Checks for CUDA availability and installs `clip` and `umap-learn`.
2. **Dataset Configuration**: Ensure that dataset archives (`.zip` or `.tar.gz`) and JSON split files are placed in `../Datasets/`. Run the dataset extraction cell block, configuring the `DATASET_NAME` variable (e.g., `'caltech101'`, `'ucf101'`, etc.) to switch between different domains.
3. **Core Modules**: Cells detailing the network classes for standard multi-modal projection, CoCoOp text-context conditioning, and ProMIM masked modeling layers.
4. **Training & Few-Shot Evaluation**: Launch experiments. The code will automatically query for existing checkpoints inside `../Extracted_Datasets/[dataset_name]/checkpoints/` before initiating training from scratch.
5. **Visualization Generation**: Execute the trailing sections to render attention heatmaps and UMAP clustering spaces corresponding to the loaded/trained models.


## Data Augmentation

Training and test transforms strictly follow the official CoOp/Dassl framework augmentation pipeline:

**Training** (applied to all train splits):
- `RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=BICUBIC)`
- `RandomHorizontalFlip()`
- `Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))`

**Testing** (applied to val/test splits):
- CLIP's default `preprocess` pipeline: `Resize → CenterCrop(224) → Normalize`

This matches the updated Dassl `transforms.py` pipeline (`random_resized_crop` + `random_flip` + `normalize`) introduced in October 2021 to align with OpenAI's original CLIP preprocessing — replacing the older `random_flip + random_translation + center_crop` pipeline.

References:
- [Dassl transforms.py](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/transforms/transforms.py)
- [CoCoOp trainer](https://github.com/KaiyangZhou/CoOp/blob/main/trainers/cocoop.py)
- [CoOp Issue #8 — transform update](https://github.com/KaiyangZhou/CoOp/issues/8)


## Evaluation Schema

The primary output results map standard top-1 classification accuracies against the Test suite of the chosen dataset across varied few-shot configurations (1, 4, 8, 16). 
The evaluation strictly enforces the standard Base-to-New Generalization protocol (CoOp benchmark). 
- **Context Vector Anchoring**: Prompt learners dynamically instantiate over `ctx_init="a photo of a"`, restricting the architecture from treating structural semantics across unseen sets as entirely latent random noise.
- **Logit Bounding**: When evaluating Zero-Shot, CoCoOp, or ProMIM against the mapped base/new testing splits, model logits are properly bounded and evaluated exclusively among their respective subset label pool combinations to ensure absolute consistency with base-to-new benchmarks published in literature.
- **Pre-computed Multi-Method Mapping**: Zero-Shot, CoCoOp, ProMIM MIMO, and ProMIM full distributions are natively extracted from cached Harmonic Mean matrices for visual analysis without imposing redundant training loops.
