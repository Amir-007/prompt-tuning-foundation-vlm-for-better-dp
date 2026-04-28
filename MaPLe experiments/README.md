# MaPLe: Multi-modal Prompt Learning Experiments

This directory contains the implementation and experiments for **MaPLe (Multi-modal Prompt Learning)** based on the CVPR 2023 paper: [MaPLe: Multi-modal Prompt Learning](https://arxiv.org/abs/2210.03117).

## Overview

MaPLe extends CoOp by introducing **deep multi-modal prompts** that are injected into both the vision and text encoders across multiple transformer layers, significantly improving generalization to unseen classes.

### Key Innovations over CoOp

| Feature | CoOp | MaPLe |
|---------|------|-------|
| Prompt Location | Text encoder only (Layer 0) | Both vision & text encoders |
| Prompt Depth | Shallow (single layer) | Deep (multiple layers, default: 9) |
| Cross-modal Coupling | None | V-L Coupler projects text → vision |
| Context Length | 16 tokens | 2 tokens (shorter but deeper) |
| Generalization | Struggles on new classes | Strong base-to-new generalization |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MaPLe Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     V-L Coupler      ┌─────────────┐          │
│  │ Text Prompt │ ──────────────────►  │Vision Prompt│          │
│  │   [V₁,V₂]   │     (Linear Proj)    │   [P₁,P₂]   │          │
│  └──────┬──────┘                      └──────┬──────┘          │
│         │                                    │                  │
│         ▼                                    ▼                  │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │Text Encoder  │                    │Vision Encoder│          │
│  │  Layer 0     │◄── Text Ctx ──────►│   Layer 0    │          │
│  │  Layer 1     │◄── Deep Ctx ──────►│   Layer 1    │          │
│  │    ...       │◄── Deep Ctx ──────►│     ...      │          │
│  │  Layer J-1   │◄── Deep Ctx ──────►│   Layer J-1  │          │
│  │  Layer J...  │                    │   Layer J... │          │
│  └──────┬──────┘                     └──────┬──────┘          │
│         │                                    │                  │
│         ▼                                    ▼                  │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │Text Features │◄── Cosine Sim ────►│Image Features│          │
│  └──────────────┘                    └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
MaPLe experiments/
├── MaPLe_Experiments.ipynb # Main experiments notebook (self-contained)
├── README.md               # This file
└── results/
    ├── History_And_Csv/    # Training histories and CSV results
    │   ├── all_results.csv
    │   ├── base_to_new_results.csv
    │   ├── zero_shot_baseline.csv
    │   └── history_*.json
    ├── plots/              # Visualizations
    │   ├── shots_comparison.png
    │   ├── performance_heatmap.png
    │   ├── base_to_new_comparison.png
    │   ├── maple_vs_coop_comparison.png
    │   ├── training_curves.png
    │   └── radar_chart.png
    └── weights/            # Trained model weights
        └── *.pt
```

## Installation

```bash
# Create environment
conda create -n maple python=3.10 -y
conda activate maple

# Install dependencies
pip install torch torchvision clip-by-openai tqdm matplotlib seaborn pandas scikit-learn
```

## Quick Start

Open `MaPLe_Experiments.ipynb` and run all cells. The notebook is self-contained with all model definitions, dataset loaders, and training code.

Make sure to update the data path in the Config class:

```python
# In the Config class
DATA_ROOT = "/path/to/your/datasets"  # Update this
```

The notebook includes:
- **Section 1**: Setup and Configuration
- **Section 2**: MaPLe Model Architecture (MultiModalPromptLearner, VLCoupler, CustomCLIP)
- **Section 3**: Dataset Loaders with base-to-new splits
- **Section 4**: Training utilities with cosine scheduler and warmup
- **Section 5**: Main experiments (few-shot learning, base-to-new generalization)
- **Section 6**: Visualization and analysis

## Experiments

### 1. Standard Few-shot Classification

Train MaPLe with different numbers of labeled examples per class:
- **1-shot**, **2-shot**, **4-shot**, **8-shot**, **16-shot**

### 2. Base-to-New Generalization (Key Evaluation)

This is the most important metric for MaPLe:

1. **Split classes**: Divide dataset classes into Base (50%) and New (50%)
2. **Train**: Only on Base classes
3. **Evaluate**: On both Base and New classes separately
4. **Report**: Base Acc, New Acc, and **Harmonic Mean (H-Mean)**

### 3. Ablation Studies

- **Prompt Depth**: Effect of J (1, 3, 6, 9, 12 layers)
- **Context Length**: Effect of n_ctx (1, 2, 4, 8 tokens)
- **V-L Coupling**: With vs without vision prompts

## Hyperparameters

| Parameter | MaPLe (Paper) | Our Default |
|-----------|---------------|-------------|
| Learning Rate | 0.0035 | 0.0035 |
| Context Length (n_ctx) | 2 | 2 |
| Prompt Depth (J) | 9 | 9 |
| Optimizer | SGD | SGD |
| Momentum | 0.9 | 0.9 |
| Weight Decay | 5e-4 | 5e-4 |
| LR Schedule | Cosine + Warmup | Cosine + Warmup |
| Batch Size | 32 | 32 |
| Epochs (16-shot) | 200 | 200 |

### Epoch Schedule by Shots
| Shots | Epochs |
|-------|--------|
| 1 | 50 |
| 4 | 100 |
| 8 | 100 |
| 16 | 200 |

## Supported Datasets

| Dataset | Classes | Description |
|---------|---------|-------------|
| EuroSAT | 10 | Satellite images |
| DTD | 47 | Describable textures |
| Caltech101 | 100 | Object categories |
| OxfordPets | 37 | Pet breeds |
| StanfordCars | 196 | Car models |
| Flowers102 | 102 | Flower species |
| Food101 | 101 | Food categories |
| FGVCAircraft | 100 | Aircraft variants |
| SUN397 | 397 | Scene categories |
| UCF101 | 101 | Action recognition |

## Results Format

### all_results.csv
```csv
dataset,num_shots,seed,n_ctx,prompt_depth,final_acc,best_val_acc,best_epoch
eurosat,16,1,2,9,92.35,91.80,180
dtd,16,1,2,9,68.42,67.95,195
...
```

### base_to_new_results.csv
```csv
dataset,num_shots,seed,base_acc,new_acc,h_mean,n_base_classes,n_new_classes
eurosat,16,1,95.20,78.45,86.01,5,5
dtd,16,1,72.30,58.60,64.74,23,24
...
```

## Comparison with CoOp

The notebook automatically generates comparison plots if CoOp results are available. Key insights:

1. **Few-shot Performance**: MaPLe typically matches or exceeds CoOp
2. **Base-to-New Generalization**: MaPLe significantly outperforms CoOp on unseen classes
3. **Training Efficiency**: MaPLe uses shorter context (2 vs 16) but deeper integration

## References

```bibtex
@inproceedings{khattak2023maple,
  title={MaPLe: Multi-modal Prompt Learning},
  author={Khattak, Muhammad Uzair and Rasheed, Hanoona and Maaz, Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19113--19122},
  year={2023}
}

@inproceedings{zhou2022coop,
  title={Learning to Prompt for Vision-Language Models},
  author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  booktitle={International Journal of Computer Vision},
  year={2022}
}
```

## License

This implementation is for research purposes. Please cite the original MaPLe paper if you use this code.
