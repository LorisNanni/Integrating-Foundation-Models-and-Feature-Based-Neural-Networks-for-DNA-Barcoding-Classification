# 3D Chaos Game Representation (3D-CGR) with Vision Transformers

This repository contains code for classifying DNA sequences using 3D Chaos Game Representation (3D-CGR) projections combined with Vision Transformers (ViT). 

The approach converts DNA sequences into 3D-CGR, extracts 2D projections (xy, yz, xz), and feeds these images into a late-fusion Vision Transformer for classification.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <REPOSITORY_LINK>
   cd 3dcgr-vit
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install torch torchvision transformers timm datasets scikit-learn matplotlib biopython scipy
   ```

3. **Data Preparation:**
   By default, the scripts expect a `data/` folder in the root directory containing the dataset files. Ensure your datasets are formatted correctly and placed in `data/`.

## Usage

The code provides an easy-to-use Command Line Interface (CLI) using `argparse`. You do not need to modify the Python scripts to change configurations.

### 1. Training (`train.py`)

Run the `train.py` script to train the model on a specific dataset.

```bash
python train.py --dataset_name <DATASET_NAME> [OPTIONS]
```

**Common Examples:**
```bash
# Train on the "fish" dataset
python train.py --dataset_name fish

# Train on the "beetle" dataset, generate and save 2D projection images
python train.py --dataset_name beetle --generate_images

# Train on a simulated dataset with custom batch size and epochs
python train.py --dataset_name "GNe50000 Filtered" --batch_size 8 --epochs 50
```

### 2. Evaluating Models (`evaluate.py`)

Run the `evaluate.py` script to evaluate the trained models and optionally produce/save logits.

```bash
python evaluate.py --dataset_name <DATASET_NAME> [OPTIONS]
```

**Common Examples:**
```bash
# Evaluate on the "fish" dataset using default inferred model path and save logits
python evaluate.py --dataset_name fish --save_logits

# Evaluate with a specific model checkpoint
python evaluate.py --dataset_name beetle --model_path path/to/model.pt
```

### Available Arguments

Both `train.py` and `evaluate.py` share the following core arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset_name` | **Required.** Dataset name to use (e.g.: `fish`, `beetle`, `birds`, `GNe50000 Filtered`) | None |
| `--generate_images`| Flag to generate and save the dot images to the `images_path` | `False` |
| `--save_logits` | Flag to save the logits in memory on the test set | `False` |
| `--data_path` | Path to the data folder containing the datasets | `data` |
| `--images_path` | Path to save the generated images | `images` |
| `--model_name` | ViT model to use from HuggingFace/timm | `timm/vit_large_patch32_224.orig_in21k` |
| `--batch_size` | Per-device train batch size | `4` |
| `--gradient_accumulation_steps` | Gradient accumulation steps | `16` |
| `--epochs` | Number of training epochs | `100` |
| `--learning_rate`| Initial learning rate | `2e-4` |

## Model Architecture

The `ViTLateFusion` model uses three input views (XY, YZ, XZ projections of the 3D-CGR). It processes each view using a shared Vision Transformer, concatenates their features, and classifies the fused representation using a multi-layer perceptron (MLP).

## License

[Add License Information Here]
