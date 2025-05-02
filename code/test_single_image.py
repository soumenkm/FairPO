# predict_single.py
import torch
import wandb # Keep import if needed, but logic removed for simplicity
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import time
import sys
import types # Used to update config namespace
from PIL import Image
from transformers import AutoFeatureExtractor # To preprocess the image

# Assuming models.py and dataset.py are in the same directory or accessible via PYTHONPATH
from models import VisionModelForCLS
# We still need COCODatasetOnDemand temporarily to get label names
from dataset import COCODatasetOnDemand

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for potential reproducibility if needed (less critical for single inference)
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

def predict_single_image(args):
    """Loads a model and predicts labels for a single image."""
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id is not None else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Checkpoint ---
    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        logging.error(f"Checkpoint file not found at: {checkpoint_path}")
        sys.exit(1)
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    # Load config and weights. Load on CPU first is safer.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # --- Get Config from Checkpoint ---
    if 'config' not in checkpoint:
        logging.error("Checkpoint does not contain 'config'. Cannot proceed.")
        sys.exit(1)
    config = types.SimpleNamespace(**checkpoint['config'])
    logging.info("Configuration loaded from checkpoint.")
    # Log the essential config loaded
    logging.info(f"  Model Name: {config.model_name}")
    logging.info(f"  Num Labels (expected): {config.num_labels}")
    logging.info(f"  Is Ref Training: {config.is_ref_training}")

    # --- Get Label Names (Workaround) ---
    # Instantiate dataset briefly just to get label names and indices map
    # Use COCO root from config if available, otherwise use a default/dummy path
    # Ensure the specified coco_root exists if needed by the dataset init
    coco_root_for_labels = getattr(config, 'coco_root', args.coco_root_for_labels)
    index_dir_for_labels = Path(coco_root_for_labels) / '.index_cache'
    try:
        logging.info(f"Attempting to load label names using COCO root: {coco_root_for_labels}")
        # Note: This might still require the dataset files/structure to exist
        #       depending on how COCODatasetOnDemand is implemented.
        #       If it fails, you might need to save label_names in the checkpoint
        #       during training.
        temp_dataset = COCODatasetOnDemand(
            root_dir=coco_root_for_labels,
            frac=0.001, # Minimal fraction
            split_name="val", # Any split should work for label names
            privileged_indices_set=set(), # Not relevant for getting names
            seed=42,
            index_dir=index_dir_for_labels,
            force_regenerate=False
        )
        label_names = temp_dataset.get_label_names()
        num_labels = len(label_names)
        # Get privileged indices if they are strictly needed by model init
        privileged_indices = set(map(int, config.privileged_indices.split(','))) if hasattr(config, 'privileged_indices') and config.privileged_indices else set()
        non_privileged_indices = list(set(range(num_labels)) - privileged_indices)
        privileged_indices = list(privileged_indices) # Model might expect lists

        logging.info(f"Successfully loaded {num_labels} label names.")
        # Verify consistency
        if num_labels != config.num_labels:
             logging.warning(f"Number of labels from dataset ({num_labels}) doesn't match config ({config.num_labels}). Using value from dataset.")
             config.num_labels = num_labels # Prioritize dataset value if different
    except Exception as e:
        logging.error(f"Failed to instantiate COCODatasetOnDemand to get label names: {e}")
        logging.error("Please ensure the COCO dataset (or at least its annotation structure) is accessible at the specified path, or modify the script to load labels differently (e.g., from a saved file or hardcoded list).")
        # Example: Hardcode if necessary as a fallback
        # label_names = ['person', 'bicycle', ..., 'toothbrush'] # List all 80 COCO names
        # num_labels = len(label_names)
        # if num_labels != config.num_labels: ... # handle potential mismatch
        sys.exit(1)


    # --- Initialize Model ---
    logging.info(f"Initializing model: {config.model_name}")
    # Ensure necessary parameters are present in config or provide defaults
    beta = getattr(config, 'beta', 2.0) # Provide default if missing
    epsilon = getattr(config, 'epsilon', 0.1) # Provide default if missing

    model = VisionModelForCLS(
        device=device, # Pass the actual device object
        model_name=config.model_name,
        num_labels=num_labels, # Use num_labels derived from dataset/names
        ref_cls_weights_path=None, # Loaded from checkpoint
        privileged_indices=privileged_indices, # Pass the lists
        non_privileged_indices=non_privileged_indices, # Pass the lists
        is_ref=config.is_ref_training,
        beta=beta,
        epsilon=epsilon,
        quant_config=None # Assuming no quantization for prediction
    ).to(device)
    logging.info("Model structure initialized.")

    # --- Load Model Weights ---
    logging.info("Loading model weights...")
    if 'model_state_dict' not in checkpoint:
        logging.error("Checkpoint does not contain 'model_state_dict'. Cannot load weights.")
        sys.exit(1)
    model_state_dict = checkpoint['model_state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if missing_keys:
        logging.warning(f"Missing keys when loading weights: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys when loading weights: {unexpected_keys}")
    logging.info("Model weights loaded successfully.")
    model.eval() # Set model to evaluation mode

    # --- Load Feature Extractor ---
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)
        logging.info(f"Loaded feature extractor for {config.model_name}")
    except Exception as e:
        logging.error(f"Failed to load feature extractor for {config.model_name}: {e}")
        logging.error("Make sure the model name in the config is a valid Hugging Face model identifier.")
        sys.exit(1)

    # --- Load and Preprocess Image ---
    image_path = Path(args.image_path).resolve()
    if not image_path.is_file():
        logging.error(f"Image file not found at: {image_path}")
        sys.exit(1)

    try:
        logging.info(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB") # Ensure image is RGB

        # Preprocess using the feature extractor
        inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        logging.info(f"Image preprocessed and moved to device. Tensor shape: {pixel_values.shape}")

    except Exception as e:
        logging.error(f"Failed to load or preprocess image: {e}")
        sys.exit(1)

    # --- Perform Inference ---
    logging.info("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        # Pass only pixel_values for inference
        outputs = model(pixels=pixel_values)

    end_time = time.time()
    logging.info(f"Inference completed in {end_time - start_time:.3f} seconds.")

    # --- Process Output ---
    # Output structure might depend on your VisionModelForCLS implementation.
    # Usually, it returns logits. Check the model's forward method return value.
    if 'logits' not in outputs:
         logging.error(f"Model output dictionary does not contain 'logits'. Available keys: {outputs.keys()}")
         # Try common alternative keys or inspect the output_dict structure
         if 'outputs' in outputs and hasattr(outputs['outputs'], 'logits'):
             logits = outputs['outputs'].logits
             logging.info("Found logits under outputs['outputs'].logits")
         elif 'preds' in outputs:
             logits = outputs['preds']
             logging.info("Found predictions under 'preds' key, assuming these are logits.")
         else:
             logging.error("Cannot determine logits from model output. Please inspect the VisionModelForCLS forward method.")
             print("Model Output:", outputs) # Print the raw output for debugging
             sys.exit(1)
    else:
        logits = outputs['logits']
        logging.info("Found logits under 'logits' key.")


    # Logits shape should be (batch_size, num_labels), here batch_size is 1
    logging.info(f"Logits shape: {logits.shape}")

    # Apply sigmoid to get probabilities (for multi-label classification)
    probabilities = torch.sigmoid(logits).squeeze() # Remove batch dimension

    # Get predicted labels based on threshold
    predicted_indices = torch.where(probabilities > args.threshold)[0].cpu().numpy()
    predicted_label_names = [label_names[i] for i in predicted_indices]

    logging.info(f"--- Prediction Results (Threshold: {args.threshold}) ---")
    if predicted_label_names:
        logging.info(f"Predicted Labels: {', '.join(predicted_label_names)}")
    else:
        logging.info("No labels predicted above the threshold.")

    # Optional: Print top N probabilities
    if args.top_n > 0:
        top_probs, top_indices = torch.topk(probabilities, args.top_n)
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        logging.info(f"--- Top {args.top_n} Probabilities ---")
        for i in range(args.top_n):
            label_name = label_names[top_indices[i]]
            prob = top_probs[i]
            logging.info(f"  {label_name}: {prob:.4f}")
    logging.info("------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Predict labels for a single image using a trained model.")

    # --- Determine User Dir (for default COCO root if needed) ---
    if "raid" in str(Path.cwd()).lower(): user_dir = "/raid/speech/soumen"
    elif "home" in str(Path.cwd()).lower(): user_dir = "/home/soumen"
    else: user_dir = "."; logging.warning(f"Defaulting user_dir to '.'")
    default_coco_root = f"{user_dir}/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014"


    # --- Required Arguments ---
    parser.add_argument('--checkpoint_path', type=str, default="output/ckpt/fairpo_model/FairPO-train/FairPO_ep5_lr5e-05_eta0.0001_eps0.1_beta2.0/checkpoint_epoch_5.pth", required=False, help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--image_path', type=str,default="/home/soumen/OML/FairPO/code/test.jpg", required=False, help='Path to the single image file for prediction')

    # --- Optional Arguments ---
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for sigmoid probability to classify a label as present')
    parser.add_argument('--top_n', type=int, default=5, help='Show top N probabilities (0 to disable)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (e.g., 0). Uses CPU if not specified or CUDA not available.')
    parser.add_argument('--coco_root_for_labels', type=str, default=default_coco_root, help='Path to COCO dataset root, needed *only* to extract label names if not stored elsewhere.')


    args = parser.parse_args()

    # Set CUDA device based on argument
    if args.gpu_id is not None and torch.cuda.is_available():
         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
         logging.info(f"Setting CUDA_VISIBLE_DEVICES to {args.gpu_id}")
    elif args.gpu_id is not None:
         logging.warning(f"GPU {args.gpu_id} requested, but CUDA is not available. Using CPU.")


    predict_single_image(args)

if __name__ == '__main__':
    # Remove the hardcoded CUDA device setting from the original script's main guard
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" # Remove this line
    main()