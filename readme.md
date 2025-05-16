# FairPO: Robust Preference Optimization for Fair Multi-Label Learning

This repository contains the official PyTorch implementation for the paper "FairPO: Robust Preference Optimization for Fair Multi-Label Learning". FairPO is a novel framework that integrates preference-based loss formulations with group-robust optimization to improve fairness in multi-label classification (MLC), particularly targeting underperforming and sensitive label groups.

## Table of Contents

1.  [Overview](#overview)
2.  [Framework](#framework)
3.  [Installation](#installation)
    *   [Prerequisites](#prerequisites)
    *   [Environment Setup](#environment-setup)
    *   [Datasets](#datasets)
4.  [Usage](#usage)
    *   [Training a Reference (SFT) Model](#training-a-reference-sft-model)
    *   [Training a FairPO Model](#training-a-fairpo-model)
    *   [Running Ablation Studies](#running-ablation-studies)
    *   [Key Arguments](#key-arguments)
5.  [File Structure](#file-structure)
6.  [Results](#results)
7.  [Citation](#citation)
8.  [License](#license)

## Overview

Multi-label classification (MLC) models often exhibit performance disparities, especially for infrequent or sensitive label categories. FairPO addresses this by:

1.  **Label Partitioning**: Dividing labels into a "privileged" set (requiring enhanced performance) and a "non-privileged" set (maintaining baseline performance).
2.  **Preference-based Loss for Privileged Labels**:
    *   If "confusing examples" exist (e.g., a true positive label score is lower than a negative label score), a preference loss (inspired by DPO, SimPO, or CPO) is applied. This encourages the model to score true positives significantly higher than their confusing negatives, and true negatives significantly lower than their confusing positives.
    *   If no confusing examples exist for a privileged label, a standard Binary Cross-Entropy (BCE) loss is used for stability.
3.  **Constrained Objective for Non-Privileged Labels**: A hinge loss ensures that the BCE loss for non-privileged labels does not degrade substantially below that of a pre-trained reference model.
4.  **Group Robust Preference Optimization (GRPO)**: A minimax optimization adaptively balances the objectives for the privileged and non-privileged groups, mitigating bias and preventing performance degradation in one group for gains in another. The alpha weights for GRPO are updated using a scaled loss mechanism for stability.

## Framework

The core idea is to formulate the learning problem as:

```
min_{W} max_{alpha_P, alpha_NP} [ alpha_P * L_Privileged(W) + alpha_NP * L_NonPrivileged(W)]
```

where:
*   `W` are the model parameters.
*   `alpha_P` and `alpha_NP` are adaptive weights for the privileged and non-privileged groups.
*   `L_Privileged` is the conditional preference-based loss (or BCE) for privileged labels.
*   `L_NonPrivileged` is the constrained BCE loss for non-privileged labels.

## Installation

### Prerequisites

*   Python 3.8+
*   PyTorch 1.10+ (with CUDA support for GPU training)
*   Transformers (Hugging Face)
*   Datasets (Hugging Face)
*   scikit-learn
*   WandB (optional, for experiment tracking)
*   Access to MS-COCO 2014 and/or NUS-WIDE datasets.

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/FairPO.git
    cd FairPO
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv fairpo_env
    source fairpo_env/bin/activate  # On Windows: fairpo_env\\Scripts\\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Datasets

*   **MS-COCO 2014**:
    *   Download the dataset from [COCO dataset](https://cocodataset.org/#download). You'll need:
        *   2014 Train images
        *   2014 Val images
        *   2014 Train/Val annotations (though `COCODatasetOnDemand` as provided uses YOLO-style label files)
    *   The `COCODatasetOnDemand` class, as used in the provided `dataset.py`, expects a specific structure, typically associated with YOLO-format labels for object detection datasets repurposed for multi-label classification:
        ```
        <coco_root>/
            images/
                train2014/
                    COCO_train2014_000000000009.jpg
                    ...
                val2014/
                    COCO_val2014_000000000042.jpg
                    ...
            labels/  <-- YOLO format: <class_id> <x_center> <y_center> <width> <height> per line
                train2014/
                    COCO_train2014_000000000009.txt
                    ...
                val2014/
                    COCO_val2014_000000000042.txt
                    ...
            coco.names      <-- File with one class name per line, matching class_ids
            .coco_index_cache/ <-- Will be created automatically by dataset.py
        ```
    *   The `labels/*.txt` files are crucial. Each line represents an object, and the first number is the `class_id`. The script extracts all unique `class_id`s from these files for an image to form its multi-label ground truth.
*   **NUS-WIDE**:
    *   Download from the [official NUS-WIDE page](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) or a mirror. You'll typically need:
        *   `Flickr` (image folder containing all images, often further subdivided)
        *   `Groundtruth/TagRelevant` (label files like `Labels_Train.txt`, `Labels_Test.txt`)
        *   `ImageList` (image path lists like `TrainImagelist.txt`, `TestImagelist.txt`)
        *   `NUS_WID_Tags` (concept/tag lists like `Concepts81.txt` or `All_Tags.txt`)
    *   Organize them under a `<nus_root>` directory as expected by `NUSWIDEDatasetOnDemand`:
        ```
        <nus_root>/
            Flickr/                 # Contains all NUS-WIDE images
                animal/
                    animal_0001.jpg
                    ...
                person/
                    person_0001.jpg
                    ...
            Groundtruth/
                Labels_Train.txt    # Space-separated binary vectors (0 or 1 for each of 81 concepts)
                Labels_Test.txt
            ImageList/
                TrainImagelist.txt  # Relative paths to images, e.g., animal\\animal_0001.jpg
                TestImagelist.txt
            NUS_WID_Tags/
                Concepts81.txt      # List of 81 concept names, one per line
            .nuswide_index_cache/   <-- Will be created automatically by dataset.py
        ```

The scripts will automatically generate index files (`.csv`) for faster loading on subsequent runs. These are stored in `.coco_index_cache` or `.nuswide_index_cache` within the respective dataset root directories by default, or in the directory specified by `--index_dir`.

## Usage

All training and ablation runs are managed via Python scripts (`train.py`, `ablation.py`).

**Important**:
*   Set `CUDA_VISIBLE_DEVICES` environment variable if you want to specify particular GPUs. E.g., `export CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=0,1 python ...`
*   The `--ref_cls_weights_path` argument is crucial. For FairPO training or ablations, it **must** point to the checkpoint of a previously trained SFT (Supervised Fine-Tuning) model on the **same dataset and with the same base Vision Transformer model**.

### Training a Reference (SFT) Model

This model serves as the baseline and provides the reference scores for FairPO.

```bash
python train.py \\
    --dataset_name coco \\
    --coco_root /path/to/your/coco2014 \\
    --is_ref_training \\
    --model_name google/vit-base-patch16-224 \\
    --epochs 25 \\
    --batch_size 32 \\
    --learning_rate 1e-4 \\
    --privileged_indices "78,70,12,21,76,79,52,54,68,18,31,47,51,10,11,49,64,22,50,19" \\
    --wandb_project "FairPO-SFT-Runs" \\
    --run_name_base "SFT_COCO_ViTBase_ep25" \\
    --checkpoint_dir ./output/sft_checkpoints 
    # For NUS-WIDE:
    # --dataset_name nuswide --nus_root /path/to/nuswide 
    # --privileged_indices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" (example head 16 for ~20% of 81 labels)
```

The best SFT model checkpoint (e.g., `checkpoint_best.pth`) saved in the specified checkpoint directory (e.g., `./output/sft_checkpoints/FairPO-SFT-Runs/SFT_COCO_ViTBase_ep25/checkpoint_best.pth`) will be used as the `--ref_cls_weights_path` for subsequent FairPO training.

### Training a FairPO Model

Once you have a trained SFT model:

```bash
python train.py \\
    --dataset_name coco \\
    --coco_root /path/to/your/coco2014 \\
    --ref_cls_weights_path ./output/sft_checkpoints/FairPO-SFT-Runs/SFT_COCO_ViTBase_ep25/checkpoint_best.pth \\
    --loss_type cpo \\
    --model_name google/vit-base-patch16-224 \\
    --epochs 25 \\
    --batch_size 32 \\
    --learning_rate 1e-4 \\
    --beta 0.5 \\
    --epsilon 0.05 \\
    --eta_alpha 0.05 \\
    --ema_decay 0.9 \\
    --delta_scaling 1e-6 \\
    --privileged_indices "78,70,12,21,76,79,52,54,68,18,31,47,51,10,11,49,64,22,50,19" \\
    --wandb_project "FairPO-Main-Runs" \\
    --run_name_base "FairPO_CPO_COCO_ViTBase_ep25" \\
    --checkpoint_dir ./output/fairpo_checkpoints
    # For NUS-WIDE, adjust --dataset_name, --nus_root, --ref_cls_weights_path, and --privileged_indices
```
*   `loss_type` can be `dpo`, `simpo`, or `cpo`.
*   Ensure `--ref_cls_weights_path` points to the correct SFT model checkpoint.

### Running Ablation Studies

The `ablation.py` script allows testing FairPO with certain components disabled. It **requires** a pre-trained SFT model specified via `--ref_cls_weights_path`.

**Run all predefined ablations (for CPO base loss type on COCO):**

```bash
python ablation.py \\
    --dataset_name coco \\
    --coco_root /path/to/your/coco2014 \\
    --ref_cls_weights_path ./output/sft_checkpoints/FairPO-SFT-Runs/SFT_COCO_ViTBase_ep25/checkpoint_best.pth \\
    --loss_type cpo \\
    --run_all_ablations \\
    --epochs 15 \\
    --wandb_project "FairPO-Ablations" \\
    --checkpoint_dir ./output/ablation_checkpoints \\
    --privileged_indices "78,70,12,21,76,79,52,54,68,18,31,47,51,10,11,49,64,22,50,19" 
    # Other parameters (lr, beta, etc.) will use defaults or can be overridden
```

**Run a specific ablation (e.g., FairPO-CPO without GRPO):**

```bash
python ablation.py \\
    --dataset_name coco \\
    --coco_root /path/to/your/coco2014 \\
    --ref_cls_weights_path ./output/sft_checkpoints/FairPO-SFT-Runs/SFT_COCO_ViTBase_ep25/checkpoint_best.pth \\
    --loss_type cpo \\
    --ablation_no_grpo \\
    --epochs 15 \\
    --wandb_project "FairPO-Ablations" \\
    --run_name_base "Ablation_CPO_NoGRPO" \\
    --checkpoint_dir ./output/ablation_checkpoints \\
    --privileged_indices "78,70,12,21,76,79,52,54,68,18,31,47,51,10,11,49,64,22,50,19"
```

### Key Arguments

*   `--dataset_name`: `coco` or `nuswide`.
*   `--coco_root`, `--nus_root`: Paths to dataset root directories.
*   `--is_ref_training`: Flag to train an SFT model. If not set, trains FairPO.
*   `--ref_cls_weights_path`: Path to SFT model weights. **Required for FairPO/ablation**.
*   `--loss_type`: `dpo`, `simpo`, `cpo` for FairPO's privileged loss.
*   `--privileged_indices`: Comma-separated string of privileged label indices.
*   `--beta`: Strength of preference term in DPO/SimPO/CPO.
*   `--epsilon`: Slack for non-privileged loss constraint.
*   `--eta_alpha`: Learning rate for GRPO's alpha weights.
*   `--ema_decay`: Decay rate for EMA of losses used in scaled GRPO updates.
*   `--delta_scaling`: Small constant for stability in scaled loss denominator for GRPO.
*   `--wandb_project`: Name of your Weights & Biases project.
*   `--run_name_base` or `--run_name`: Custom name for the run. `generate` (default) creates one automatically.
*   `--no_wandb`: Disable WandB logging.
*   `--checkpoint_dir`: Directory to save model checkpoints.
*   (Ablation script specific) `--run_all_ablations`, `--ablation_no_preference_loss`, `--ablation_no_np_constraint`, `--ablation_no_grpo`.

Refer to the `argparse` sections in `train.py` and `ablation.py` for a full list of arguments and their defaults.

## File Structure

```
FairPO/
├── train.py                  # Main script for SFT and FairPO training
├── ablation.py               # Script for running ablation studies
├── models.py                 # Defines VisionModelForCLS (the FairPO model architecture)
├── dataset.py                # COCODatasetOnDemand and NUSWIDEDatasetOnDemand classes
├── metrics.py                # Loss computation and evaluation metrics logic
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── output/                   # Default directory for checkpoints and logs (created on run)
    ├── sft_checkpoints/
    ├── fairpo_checkpoints/
    └── ablation_checkpoints/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details (if you add one).
(Consider adding a `LICENSE` file to your repository.)
