# test.py
import torch
from torch.utils.data import DataLoader
import wandb
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import sys
import types # Used to update config namespace

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Assuming models.py and dataset.py are in the same directory or accessible via PYTHONPATH
from archive.models import VisionModelForCLS
from dataset import COCODatasetOnDemand, collate_fn_skip_none # Use OnDemand and collate_fn

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for potential reproducibility if needed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class Tester:
    def __init__(self, config_args):
        # Initial config from command line arguments
        self.cli_config = config_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # --- Load Checkpoint Early to Get Saved Config ---
        self.checkpoint_path = Path.cwd() / self.cli_config.checkpoint_path
        logging.info(f"Loading checkpoint config from: {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False) # Load config on CPU first

        # --- Combine CLI Args and Checkpoint Config ---
        # Create a new config object, prioritizing CLI args for execution params,
        # and checkpoint config for model/data/hyperparams used during training.
        self.config = types.SimpleNamespace(**self.checkpoint['config']) # Start with loaded config

        # Override specific parameters with CLI arguments if they are crucial for test execution
        # or environment setup.
        self.config.coco_root = self.cli_config.coco_root
        self.config.index_dir = self.cli_config.index_dir if self.cli_config.index_dir else Path(self.config.coco_root) / '.index_cache' # Handle default index dir
        self.config.test_frac = self.cli_config.test_frac   # Use CLI value for test fraction
        self.config.batch_size = self.cli_config.batch_size

        # Log the effective config being used for the test
        logging.info("--- Effective Configuration for Testing ---")
        for key, value in vars(self.config).items():
            logging.info(f"  {key}: {value}")
        logging.info("-------------------------------------------")

        # --- Load Data (Test Split) using Effective Config ---
        logging.info("Loading test dataset...")
        # Privileged indices now come from self.config (loaded from checkpoint)
        if hasattr(self.config, 'privileged_indices') and self.config.privileged_indices:
             # Assume it was saved as a comma-separated string in the checkpoint config
            privileged_indices_set = set(map(int, self.config.privileged_indices.split(',')))
        else:
            privileged_indices_set = set()
            logging.warning("No privileged indices found in checkpoint config or it was empty. Assuming empty set.")

        self.test_dataset = COCODatasetOnDemand(
            root_dir=self.config.coco_root,
            frac=self.config.test_frac,
            split_name="test",
            privileged_indices_set=privileged_indices_set,
            seed=seed, # Seed might not be in old configs, use default
            index_dir=self.config.index_dir,
            force_regenerate=False # Usually False for testing, but could be a CLI arg if needed
        )

        self.label_names = self.test_dataset.get_label_names()
        self.num_labels = len(self.label_names)
        # Get indices from the dataset instance, ensuring consistency
        self.privileged_indices = self.test_dataset.privileged_indices
        self.non_privileged_indices = self.test_dataset.non_privileged_indices

        logging.info(f"Test samples: {len(self.test_dataset)}")
        logging.info(f"Num labels: {self.num_labels}")
        logging.info(f"Privileged Indices: {self.privileged_indices}")
        logging.info(f"Non-Privileged Indices: {self.non_privileged_indices}")

        # --- Create DataLoader ---
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size, # Use potentially overridden batch size
            shuffle=False,
            num_workers=self.config.num_workers, # Use potentially overridden num_workers
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_skip_none
        )

        # --- Initialize Model using Effective Config ---
        # Model name, beta, epsilon etc., now come from self.config (loaded from checkpoint)
        logging.info(f"Initializing model: {self.config.model_name}")
        self.model = VisionModelForCLS(
            device=self.device,
            model_name=self.config.model_name,
            num_labels=self.num_labels, # Determined from loaded data
            ref_cls_weights_path=None, # Weights loaded from checkpoint's state_dict
            privileged_indices=self.privileged_indices, # Determined from loaded data
            non_privileged_indices=self.non_privileged_indices, # Determined from loaded data
            loss_type=self.config.loss_type, # Use loss_type from checkpoint
            is_ref=self.config.is_ref_training, # Check if 'is_ref_training' was saved and True
            beta=self.config.beta, # Use getattr for backwards compat if beta wasn't saved
            epsilon=self.config.epsilon, # Use getattr for backwards compat
            quant_config=None
        ).to(self.device)
        logging.info("Model structure initialized.")

        # --- Load Model Weights ---
        self._load_model_weights() # Load weights from self.checkpoint
        self.val_step = 0
        
        # --- GRPO Alpha Weights (only for FairPO testing) ---
        if not self.config.is_ref_training:
            self.alpha_privileged = self.checkpoint.get('alpha_privileged', None)
            self.alpha_non_privileged = self.checkpoint.get('alpha_non_privileged', None)

    def _load_model_weights(self):
        """Loads model weights from the pre-loaded checkpoint dictionary."""
        logging.info("Loading model weights onto initialized structure...")
        model_state_dict = self.checkpoint['model_state_dict']

        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys when loading weights: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading weights: {unexpected_keys}")
        logging.info("Model weights loaded successfully.")

    def _process_batch(self, batch: dict):
        """
        Moves batch to device, performs forward pass, and computes metrics.

        Returns:
            Tuple of (results_dict, current_batch_size) or None if batch is invalid.
            results_dict contains 'outputs', 'logits', 'loss', 'acc', 'f1', 'map', 'em'.
        """
        if batch is None or 'pixels' not in batch or batch['pixels'].numel() == 0:
            logging.warning("Skipping empty or invalid batch.")
            return None

        pixels = batch['pixels'].to(self.device)
        labels = batch['labels'].to(self.device)
        current_batch_size = pixels.size(0)

        output_dict = self.model(pixels=pixels, labels=labels)
        return output_dict, current_batch_size
    
    # --- Helper to update epoch metrics accumulator ---
    def _update_epoch_metrics(self, epoch_metrics: dict, results_dict: dict, batch_size: int):
        """Updates the epoch metrics dictionary with results from a batch."""

        # Accumulate Losses (weighted by batch size)
        loss_comp = results_dict.get('loss')
        if self.config.is_ref_training:
            epoch_metrics['loss_sft'] += loss_comp['loss'].item() * batch_size
        else:
            # For FairPO, loss_components['loss'] was placeholder sum, use priv/non_priv
            epoch_metrics['loss_priv'] += loss_comp['privileged'].item() * batch_size
            epoch_metrics['loss_non_priv'] += loss_comp['non_privileged'].item() * batch_size
            epoch_metrics['loss_total'] += (loss_comp.get('privileged') * self.alpha_privileged.item() + loss_comp.get('non_privileged') * self.alpha_non_privileged.item()) * batch_size

        # Accumulate Accuracies (weighted by batch size)
        acc_comp = results_dict.get('acc')
        epoch_metrics['acc_overall'] += acc_comp['acc'] * batch_size
        epoch_metrics['acc_priv'] += acc_comp['privileged'] * batch_size
        epoch_metrics['acc_non_priv'] += acc_comp['non_privileged'] * batch_size
        
        # Accumulate EM (weighted by batch size)
        em_comp = results_dict.get('em')
        epoch_metrics['em'] += em_comp['em'] * batch_size
        epoch_metrics['em_priv'] += em_comp['em_privileged'] * batch_size
        epoch_metrics['em_non_priv'] += em_comp['em_non_privileged'] * batch_size

        # Accumulate F1 Scores (weighted by batch size)
        f1_comp = results_dict.get('f1')
        epoch_metrics['f1'] += f1_comp['f1'] * batch_size
        epoch_metrics['f1_priv'] += f1_comp['f1_privileged'] * batch_size
        epoch_metrics['f1_non_priv'] += f1_comp['f1_non_privileged'] * batch_size

        # Accumulate mAP Scores (weighted by batch size)
        map_comp = results_dict.get('map')
        epoch_metrics['map_overall'] += map_comp['mAP'] * batch_size
        epoch_metrics['map_priv'] += map_comp['mAP_privileged'] * batch_size
        epoch_metrics['map_non_priv'] += map_comp['mAP_non_privileged'] * batch_size
    
    # --- Helper to calculate epoch averages ---
    def _calculate_epoch_averages(self, epoch_metrics: dict, processed_items: int) -> dict:
        """Calculates average metrics for an epoch."""
        avg_metrics = {}
        if processed_items == 0: # Avoid division by zero
            logging.warning("No items processed in epoch, metrics will be NaN.")
            # Populate with NaNs based on expected keys in epoch_metrics
            for k in epoch_metrics: avg_metrics[k.replace('loss_', 'avg_loss_').replace('acc_', 'avg_acc_').replace('f1_', 'avg_f1_').replace('map_', 'avg_map_').replace('em_', 'avg_em_')] = np.nan
            return avg_metrics

        for key, value in epoch_metrics.items():
            avg_key = key.replace('loss_', 'avg_loss_').replace('acc_', 'avg_acc_').replace('f1_', 'avg_f1_').replace('map_', 'avg_map_').replace('em_', 'avg_em_')
            avg_metrics[avg_key] = value / processed_items
        
        return avg_metrics
    
    # --- Validate Method ---
    def _validate(self) -> dict:
        """Runs validation loop and returns averaged metrics."""
        logging.info(f"Starting test...")
        self.model.eval()
        # Initialize accumulator dict
        val_epoch_metrics = {
            'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
            'acc_overall': 0.0, 'acc_priv': 0.0, 'acc_non_priv': 0.0,
            'em': 0.0, 'em_priv': 0.0, 'em_non_priv': 0.0,
            'f1': 0.0, 'f1_priv': 0.0, 'f1_non_priv': 0.0,
            'map_overall': 0.0, 'map_priv': 0.0, 'map_non_priv': 0.0
        }
        processed_items = 0

        with torch.no_grad():
            val_iter = tqdm(self.test_loader, desc=f"Test", unit="batch", leave=False)
            for batch in val_iter:
                processed_batch_info = self._process_batch(batch)
                if processed_batch_info is None:
                    continue
                results_dict, current_batch_size = processed_batch_info

                # Use the helper to accumulate other metrics
                self._update_epoch_metrics(val_epoch_metrics, results_dict, current_batch_size)
                processed_items += current_batch_size

        # Calculate average validation metrics
        avg_val_metrics_raw = self._calculate_epoch_averages(val_epoch_metrics, processed_items)

        # Prefix keys with 'val/'
        avg_val_metrics = {f"test/{k.replace('avg_', '')}": v for k, v in avg_val_metrics_raw.items()}

        # Increment val_step AFTER calculations for this epoch
        self.val_step += 1

        self.model.train() # Set back to training mode
        return avg_val_metrics # Return dict with 'val/' prefixes

    # --- Test Method ---
    def test(self):
        """Main testing loop."""
        # Validation
        avg_val_metrics = self._validate() # Returns dict with 'val/...' prefixed keys

        # Log epoch summary to console
        self._log_epoch_summary_console(avg_val_metrics, mode='test')

        logging.info("Testing finished.")

    # --- Helper to log validation summary (console) ---
    def _log_epoch_summary_console(self, avg_epoch_metrics: dict, mode: str):
        """Logs the summary metrics for an epoch (test) to the console."""
        if mode not in ['test']:
            logging.warning(f"Invalid mode '{mode}' provided to _log_epoch_summary_console. Skipping log.")
            return

        # Determine prefix based on mode
        prefix = f"{mode}/"
        title_prefix = "Training" if mode == 'train' else "Testing"
        step_info = f"(Train Step {self.train_step})" if mode == 'train' else f"(Test Step {self.val_step})"

        logging.info(f"--- {title_prefix} Results {step_info} ---")

        # Log Losses
        if self.config.is_ref_training:
            # SFT mode only has one loss
            logging.info(f"  Avg Loss (SFT): {avg_epoch_metrics.get(f'{prefix}loss_sft'):.4f}")
        else:
            # FairPO mode has component losses
            logging.info(f"  Avg Loss (Priv): {avg_epoch_metrics.get(f'{prefix}loss_priv'):.4f}")
            logging.info(f"  Avg Loss (Non-Priv): {avg_epoch_metrics.get(f'{prefix}loss_non_priv'):.4f}")
            # The 'loss_total' key should store the combined loss used for optimization/evaluation
            logging.info(f"  Avg Loss (Combined): {avg_epoch_metrics.get(f'{prefix}loss_total'):.4f}")

        # Log Accuracies
        logging.info(f"  Avg Acc (Overall): {avg_epoch_metrics.get(f'{prefix}acc_overall'):.4f}")
        logging.info(f"  Avg Acc (Priv): {avg_epoch_metrics.get(f'{prefix}acc_priv'):.4f}")
        logging.info(f"  Avg Acc (Non-Priv): {avg_epoch_metrics.get(f'{prefix}acc_non_priv'):.4f}")
        
        # Log Exact Match (EM) Scores
        logging.info(f"  Avg EM (Overall): {avg_epoch_metrics.get(f'{prefix}em'):.4f}")
        logging.info(f"  Avg EM (Priv): {avg_epoch_metrics.get(f'{prefix}em_priv'):.4f}")
        logging.info(f"  Avg EM (Non-Priv): {avg_epoch_metrics.get(f'{prefix}em_non_priv'):.4f}")

        # Log F1 Scores
        logging.info(f"  Avg F1 (Overall): {avg_epoch_metrics.get(f'{prefix}f1'):.4f}")
        logging.info(f"  Avg F1 (Priv): {avg_epoch_metrics.get(f'{prefix}f1_priv'):.4f}")
        logging.info(f"  Avg F1 (Non-Priv): {avg_epoch_metrics.get(f'{prefix}f1_non_priv'):.4f}")

        # Log mAP Scores
        logging.info(f"  Avg mAP (Overall): {avg_epoch_metrics.get(f'{prefix}map_overall'):.4f}")
        logging.info(f"  Avg mAP (Priv): {avg_epoch_metrics.get(f'{prefix}map_priv'):.4f}")
        logging.info(f"  Avg mAP (Non-Priv): {avg_epoch_metrics.get(f'{prefix}map_non_priv'):.4f}")

        logging.info(f"---------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Test FairPO/SFT Model for Multi-Label Classification")

    # --- Determine User Dir (as before) ---
    if "raid" in str(Path.cwd()).lower(): user_dir = "/raid/speech/user"
    elif "home" in str(Path.cwd()).lower(): user_dir = "/home/user"
    else: user_dir = "."; logging.warning(f"Defaulting user_dir to '.'")
    default_coco_root = f"{user_dir}/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014"

    # --- Required Argument ---
    ckpt_path = "output/ckpt/FairPO-train/SFT_lr5e-05_frac1.00_ep5/checkpoint_best.pth"
    parser.add_argument('--checkpoint_path', type=str, default=ckpt_path, help='Path to the model checkpoint file to test')

    # --- Essential CLI Arguments for Test Setup ---
    parser.add_argument('--coco_root', type=str, default=default_coco_root, help='Root directory of the COCO dataset (may override checkpoint config if needed)')
    parser.add_argument('--index_dir', type=str, default=None, help='Directory for dataset index files (overrides checkpoint default if provided)')
    parser.add_argument('--test_frac', type=float, default=0.1, help='Fraction of test data (overrides checkpoint default)')
    parser.add_argument('--batch_size', type=int, default=32, help='Testing batch size (overrides checkpoint default)')

    args = parser.parse_args()

    tester = Tester(args)
    tester.test()

if __name__ == '__main__':
    main()