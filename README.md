# DDA ML Pipeline

This repository contains the **Machine Learning pipeline** used to train and export models for the Dynamic Difficulty Adjustment (DDA) Unity plugin.  
It provides scripts for **training regressors** and **exporting models to ONNX** for runtime inference inside Unity via ONNX Runtime.

---

## Features
- Automated training pipeline: preprocessing and model fitting in one step.  
- Training: per-metric regressors using [XGBoost](https://xgboost.readthedocs.io).  
- Export to ONNX: convert trained models with embedded `feature_order` metadata.  
- Unity integration: exported models can be imported into Unity and executed by the MLDecisionEngine.  

---

## Quick Start

### 1. Create the environment
```bash
conda env create -f environment.yml -p .venv
conda activate ./.venv
```

### 2. Train regressors
Run the training script to generate `.pkl` models.  
The script automatically handles preprocessing.
```bash
python train_model.py --out models/
```

### 3. Export to ONNX
Convert trained models into ONNX format with feature metadata for Unity.
```bash
python export_multi_onnx.py   --models models/   --out onnx_models/   --n-features 7   --output-names Health Experience SpawnRate SpawnCount Difficulty WarmupProgress Intensity
```

---

## Unity Integration
- Copy the exported `.onnx` files into your Unity project under `Assets/DDA/MLModels/`.  
- The DDA plugin’s **ScriptedImporter** will automatically register them as `TextAsset`.  
- The **MLDecisionEngine** loads them at runtime and applies difficulty deltas based on player metrics.  

---

## Notes
- Keep `.venv/`, `.pkl` and exported ONNX models out of version control (`.gitignore`).  
- Models are **per-metric**: each ONNX file predicts the delta for a single game metric.  
- Feature ordering is stored in model metadata to ensure consistency between training and inference.  
