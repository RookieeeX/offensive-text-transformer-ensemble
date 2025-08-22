# Offensive Language Detection with Transformers (BERT & RoBERTa Ensembles)

## Overview

This repository contains code, pre-trained model checkpoints, and experiment scripts for the project:

**"Ensemble Strategies for Offensive Language Detection on the Potato-Prolific Dataset"**
Author: Xun Feng

---

## Dataset

- **Source**: [Potato-Prolific Dataset Offensiveness Subset](https://github.com/Jiaxin-Pei/Potato-Prolific-Dataset)
- **Task**: Classify English texts as 'non-offensive' (`0`) or 'offensive' (`1`).
- Only entries where `race == "White"` are used for training and evaluation to ensure statistical robustness.
- The data file should be placed at `Potato-Prolific-Dataset-main/dataset/offensiveness/raw_data.csv`.

**Label conversion:**
- If `offensiveness == 1`, label as `0` (Non-offensive)
- If `offensiveness in {2,3,4,5}`, label as `1` (Offensive)

---

## File Structure

```
.
├── requirements.txt              # Python dependencies for full replication
├── train_bert_binary_model.ipynb     # Script for BERT binary classifier training
├── train_roberta_binary_model.ipynb  # Script for RoBERTa binary classifier training
├── ensemble_model_onlyBERT.ipynb     # Ensemble inference for BERT models
├── ensemble_model_onlyRoBERTa.ipynb  # Ensemble inference for RoBERTa models
├── ensemble_all_models.ipynb         # Ensemble inference for all models (BERT + RoBERTa)
├── models/
│     ├── bert_seed42.bin
│     ├── roberta_binary_seed42.bin
│     └── ... (other model files)
└── Potato-Prolific-Dataset-main/ # Data folder (not provided in this repository)
```

---

### Model Checkpoints and Pretrained Weights

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16927866.svg)](https://doi.org/10.5281/zenodo.16927866)

**All pretrained model checkpoints for BERT and RoBERTa used in this project are available via Zenodo:**
https://doi.org/10.5281/zenodo.16927866
- Download and extract all model `.bin` files into a directory named `models/` at the root of this repository.
- The code assumes all model .bin files are present in the `models/` directory.
- **You do NOT need to retrain the models unless you want to reproduce training from scratch.**
---
## Environment & Dependencies
All code was tested under **Windows 10** with:
- Python 3.8.18
- PyTorch 1.11.0 (CUDA 11.3, cuDNN 8.2)
- transformers==4.21.0
- scikit-learn==1.0.2
- numpy==1.20.3
- pandas==1.4.1
- tqdm==4.62.3
- matplotlib==3.7.2
- seaborn==0.11.2
Full list in `requirements.txt`.  
A CUDA-enabled GPU (≥4GB VRAM) is recommended for training/evaluation.
---
## How to Reproduce Experiments

### 1. Data Preparation

Download the dataset [here](https://github.com/Jiaxin-Pei/Potato-Prolific-Dataset) and place the csv file at:
   ```
   Potato-Prolific-Dataset-main/dataset/offensiveness/raw_data.csv
   ```

### 2. Environment Setup

```bash
pip install -r requirements.txt
```
or use a suitable conda environment.

### 3. Prepare Model Checkpoints:
- Download model weights from [Zenodo](https://doi.org/10.5281/zenodo.16927866)  
- Extract all `.bin` files into the `models/` directory as described above.

### 4. Train Models (optional, if not using pretrained checkpoints):
- Run `train_bert_binary_model.ipynb` (set `RANDOM_STATE` for different seeds as needed)
- Run `train_roberta_binary_model.ipynb` (set `RANDOM_STATE` for different seeds as needed)
- Model checkpoints will be saved to `models/`

---

## Results & Report

Detailed experiment results and analysis are available in the project report (not included in this repository; please contact the author if required).

---

## Reproducibility Notes

- All random seeds, data splits, and hyperparameters are fixed in code.
- Hardware: NVIDIA RTX 3050 (4GB VRAM), AMD Ryzen 7 6800H, 16GB RAM. Lower VRAM may require smaller batch sizes.

---

## Citation

If you use this codebase or the provided models, please cite the original dataset and this project's Zenodo record:
- [Potato-Prolific Dataset](https://github.com/Jiaxin-Pei/Potato-Prolific-Dataset)
- Zenodo model archive: https://doi.org/10.5281/zenodo.16927866

---

## Contact

For questions or bug reports, please contact Xun Feng or open an issue in this repository.
