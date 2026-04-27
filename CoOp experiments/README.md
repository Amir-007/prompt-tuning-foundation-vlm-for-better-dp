# CoOp Experiments

This directory contains a notebook implementation of **CoOp (Context Optimization)** for Vision-Language Models, following Zhou et al.

## Paper Reference

> Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). **Learning to Prompt for Vision-Language Models**. International Journal of Computer Vision (IJCV).

- [Paper](https://arxiv.org/abs/2109.01134)
- [Original Code](https://github.com/KaiyangZhou/CoOp)

## Overview

CoOp learns continuous prompt vectors for CLIP instead of using hand-crafted text prompts. The learnable context vectors `[V_1, V_2, ..., V_M]` are optimized for downstream classification.

### Key Concepts

1. **Unified Context**: Same prompt vectors shared across all classes
2. **Class-Specific Context (CSC)**: Different prompt vectors for each class
3. **Token Position**: Class token at the end vs middle of the prompt

## Notebook: CoOp_Experiments.ipynb

### Quick Start (Local GPU - A4000)

1. **Update paths in cell 5** (`DATA_ROOT`, `OUTPUT_BASE`)
2. **Click "Run All"**
3. If interrupted, run again to resume from saved progress

```python
# Cell 5
DATA_ROOT = "/path/to/your/datasets"
OUTPUT_BASE = "/path/to/output/coop_results"
```

The notebook automatically:
- installs missing dependencies
- creates output folders
- runs the configured experiment sweep
- saves progress, histories, checkpoints, and model weights
- exports result tables and plots
- resumes from `experiment_progress.pkl` if interrupted

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Context tokens (`n_ctx`) | 16 |
| Learning rate | 0.002 |
| Warmup epochs | 1 |
| Warmup LR | 1e-5 |
| Batch size | 32 |
| DataLoader workers | 4 |
| CLIP backbone | ViT-B/16 |
| Prompt template | `a photo of a {}.` |
| Seed list | `[1]` |

### Epoch Schedule (Paper-aligned)

| Shots | Epochs |
|-------|--------|
| 1-shot | 50 |
| 4-shot | 100 |
| 8-shot | 200 |
| 16-shot | 200 |

### Sweep Configuration (Current Notebook)

| Parameter | Value |
|-----------|-------|
| Datasets | `eurosat`, `oxfordpets`, `dtd`, `caltech101`, `oxfordflowers`, `fgvcaircraft`, `stanfordcars`, `ucf101`, `food101` |
| Shots | `[1, 4, 8, 16]` |
| Context types | `csc=False` (Unified), `csc=True` (CSC) |
| Class token positions | `["end", "middle"]` |

### Total Experiments

```
9 datasets × 4 shots × 2 CSC variants × 2 positions × 1 seed = 144 experiments
```

## Dataset Registry (from Notebook)

| Dataset key | Folder under `DATA_ROOT` | Split JSON | Image subdirectory |
|-------------|---------------------------|------------|------------------|
| `caltech101` | `caltech-101` | `split_zhou_Caltech101.json` | `101_ObjectCategories` |
| `oxfordpets` | `oxford_pets` | `split_zhou_OxfordPets.json` | `images` |
| `dtd` | `dtd` | `split_zhou_DescribableTextures.json` | `images` |
| `eurosat` | `eurosat` | `split_zhou_EuroSAT.json` | `2750` |
| `oxfordflowers` | `oxford_flowers` | `split_zhou_OxfordFlowers.json` | `jpg` |
| `fgvcaircraft` | `fgvc_aircraft` | `split_zhou_FGVCAircraft.json` | `images` |
| `stanfordcars` | `stanford_cars` | `split_zhou_StanfordCars.json` | *(none)* |
| `ucf101` | `ucf101` | `split_zhou_UCF101.json` | `UCF-101-midframes` |
| `food101` | `food-101` | `split_zhou_Food101.json` | `images` |

## Required `DATA_ROOT` Layout

```
/your/dataset/path/
├── caltech-101/
│   ├── 101_ObjectCategories/
│   └── split_zhou_Caltech101.json
├── oxford_pets/
│   ├── images/
│   └── split_zhou_OxfordPets.json
├── dtd/
│   ├── images/
│   └── split_zhou_DescribableTextures.json
├── eurosat/
│   ├── 2750/
│   └── split_zhou_EuroSAT.json
├── oxford_flowers/
│   ├── jpg/
│   └── split_zhou_OxfordFlowers.json
├── fgvc_aircraft/
│   ├── images/
│   └── split_zhou_FGVCAircraft.json
├── stanford_cars/
│   └── split_zhou_StanfordCars.json
├── ucf101/
│   ├── UCF-101-midframes/
│   └── split_zhou_UCF101.json
└── food-101/
    ├── images/
    └── split_zhou_Food101.json
```

## Output Structure

```text
/scratch/HS400/coop_results/               # default OUTPUT_BASE, editable in cell 5
├── output/
│   ├── experiment_progress.pkl
│   ├── all_results.csv
│   ├── summary_by_variant.csv
│   ├── summary_by_dataset.csv
│   ├── zero_shot_baseline.csv
│   ├── history_{exp_key}.json
│   ├── prompt_interpretation_{dataset}.csv
│   ├── checkpoints/
│   │   └── {exp_key}_checkpoint.pt
│   └── weights/
│       └── {exp_key}_weights.pt
└── plots/
    ├── shots_comparison.png
    ├── csc_comparison.png
    ├── token_position_comparison.png
    ├── performance_heatmap.png
    ├── training_curves.png
    ├── best_accuracy_per_dataset.png
    ├── improvement_over_zero_shot.png
    ├── loss_curves.png
    ├── aggregated_bar_chart.png
    ├── lr_schedule.png
    ├── radar_chart.png
    ├── quality_analysis_{dataset}_{num_shots}shot.png
    ├── attention_maps_{dataset}.png
    ├── tsne_comparison_{dataset}.png
    ├── efficiency_tradeoff.png
    └── macro_average_accuracy.png
```

## Visualizations

| Plot | Description |
|------|-------------|
| `shots_comparison.png` | Accuracy vs shots for each dataset |
| `csc_comparison.png` | Unified vs CSC comparison |
| `token_position_comparison.png` | End vs middle token position comparison |
| `performance_heatmap.png` | Dataset × variant accuracy heatmap |
| `training_curves.png` | Validation accuracy over epochs |
| `best_accuracy_per_dataset.png` | Best variant per dataset-shot |
| `improvement_over_zero_shot.png` | CoOp improvement over zero-shot CLIP |
| `loss_curves.png` | Training vs validation loss curves |
| `aggregated_bar_chart.png` | All variants compared side-by-side |
| `lr_schedule.png` | Warmup + cosine LR schedule |
| `radar_chart.png` | 16-shot variant comparison across datasets |
| `quality_analysis_{dataset}_{num_shots}shot.png` | 6-panel quality diagnostics |
| `attention_maps_{dataset}.png` | Attention rollout visualizations |
| `tsne_comparison_{dataset}.png` | Zero-shot vs CoOp embedding-space visualization |
| `efficiency_tradeoff.png` | Accuracy vs shot-count trade-off |
| `macro_average_accuracy.png` | Macro-average accuracy across datasets |

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| `exp_key` | Unique experiment id |
| `dataset` | Dataset key |
| `num_shots` | Shot count |
| `seed` | Random seed |
| `csc` | Class-specific context flag |
| `class_token_position` | `end` or `middle` |
| `n_ctx` | Number of context tokens |
| `final_acc` | Final test accuracy |
| `best_acc` | Best observed test accuracy |
| `best_val_acc` | Best validation accuracy |
| `weights_path` | Final weights file path |
| `train_loss` | Training loss (history) |
| `val_loss` | Validation loss (history) |
| `val_acc` | Validation accuracy (history) |
| `lr` | Learning rate per epoch (history) |
| `zero_shot_acc` | Zero-shot CLIP baseline accuracy |

## Features

### Resume Capability

Progress is stored in `experiment_progress.pkl`:
- completed experiments are skipped
- interrupted runs continue from saved state

### Checkpointing and Weights

- checkpoints are saved every 25 epochs
- final prompt learner weights are saved per experiment
- temporary checkpoints are removed after successful completion

### Class Names

Class names are read directly from official CoOp split files (`split_zhou_*.json`), matching the original setup.

## Files in This Directory

| File | Description |
|------|-------------|
| `CoOp_Experiments.ipynb` | Main experiment notebook |
| `README.md` | This file |
| `results/` | Local artifacts (if created) |

## Requirements

- **GPU**: NVIDIA A4000 (or similar with 16GB+ VRAM)
- **Python**: 3.8+
- **Core dependencies** (auto-installed by notebook if missing):
  - `clip` (OpenAI CLIP)
  - `torch`, `torchvision`
  - `tqdm`, `pandas`, `matplotlib`
- **Additional analysis cells** use:
  - `seaborn`
  - `scikit-learn`

## Troubleshooting

### "Split file not found"
Ensure each dataset folder includes its matching `split_zhou_*.json`.

### "DATA_ROOT does not exist"
Update `DATA_ROOT` in cell 5 to your local dataset root.

### Interrupted training
Run the notebook again; it resumes from `experiment_progress.pkl`.

### CUDA out of memory
Reduce `batch_size` in `CONFIG` (e.g., 32 -> 16 or 8), close other GPU jobs, and rerun.
