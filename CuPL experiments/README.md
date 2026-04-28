# CuPL (Customized Prompts via Language models) - Local LLM Implementation

This sub-directory contains version 11 (`v11 fix loading`) of the CuPL zero-shot pipeline, heavily refactored to prioritize local, self-contained execution and scalable data organization.

## Key Changes & Refactoring

1. **Removed Gemini API Dependency**:
   - The previous implementation relied on Google's Gemini API which frequently encountered rate limits and required manual API key setups. 
   - Gemini generation, API keys, and rate-limiting wait loops have been completely removed.
   
2. **Default Local LLM Engines (No UI Prompts)**:
   - We now rely gracefully on local fallback engines natively, such as **TinyLlama-1.1B**, **Phi-2**, or **Mistral-7B** via Hugging Face `transformers` as our default generation backend.
   - It runs completely locally within your GPU instance. TinyLlama sits at ~1 GB footprint, sharing VRAM flawlessly alongside models like `ViT-B/16` CLIP.
   - Any ipywidgets dropdown menus have been stripped; the dictionary configurations dynamically unpack based on the `ACTIVE_LLM_KEY` variable to ensure zero user blocking / interactions required for fully automated pipeline runs.

3. **Standardized Dataset Loading**:
   - Integrated the robust, centralized dataset loaders from the CoCoOp / ProMIM benchmarks (`GenericJSONSplitDataset` and `DATASET_CONFIGS`).
   - Legacy independent dataset wrapper classes (`OxfordPetsDataset`, `Flowers102Dataset`, etc.) have been completely sunset.
   - Replaced redundant directory traversal logic with elegant and centralized dataset dictionaries. Handles `.zip` extractions, nested folders, and custom JSON splits smoothly.
   - You only configure `DATASET_NAME` in a single place to pivot uniformly between tasks (e.g. `ucf101` vs `eurosat`).
   - **Split correctness fix**: `load_single_dataset()` now returns all three splits separately — `train_dataset`, `val_dataset`, `test_dataset` — rather than silently merging val into train. This ensures `val_loader` and `test_loader` evaluate on distinct, non-overlapping subsets.
   - `full_dataset` is  built as `ConcatDataset([train_dataset, val_dataset, test_dataset])` (all images). This is used for the embedding-space visualisations (PCA / t-SNE / UMAP) so they cover the full data distribution.

4. **Dynamic Output Organization**:
   - The entire saving logic scaling out JSON prompt files and validation plots has been revamped to use `f"{DATASET_NAME}"` automatically.
   - Prompt `.json` files are automatically dumped and loaded distinctly inside `{DATASET_NAME}_cupl_prompts/`.
   - Resulting plots natively save under `{DATASET_NAME}/<chart-name>.png`, preventing clutter and preventing experiments from overwriting each other in the root workspace.

## Expected Directory Requirements

To run the unified `GenericJSONSplitDataset` functionality correctly, make sure your environment contains the required structural shortcuts initialized according to notebook Step 3/3b.

## Execution Sequence

When executing `EEEM068_Merged_ZeroShot_CuPL.ipynb`:
1. Environment Setup triggers CoCoOp JSON Loading standards natively.
2. Calls `load_single_dataset(DATASET_NAME)` to extract, split, and wrap the dataset — prints `train / val / test` sizes on success.
3. Builds `full_dataset` as `ConcatDataset([train, val, test])` for complete embedding visualisation coverage.
4. Automatically caches your targeted parameter `ACTIVE_LLM_KEY` via `local_generate()`.
5. Dispatches prompt generation batches (typically 10 calls per template) natively.
6. Saves visual analytics and evaluation benchmarks safely isolated inside dataset-specific folders.
