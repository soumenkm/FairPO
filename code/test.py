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
from models import VisionModelForCLS
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
        self.config.wandb_project = self.cli_config.wandb_project # Use CLI value for logging destination
        self.config.is_wandb = self.cli_config.is_wandb         # Use CLI value for enabling/disabling logging
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
            is_ref=self.config.is_ref_training, # Check if 'is_ref_training' was saved and True
            beta=self.config.beta, # Use getattr for backwards compat if beta wasn't saved
            epsilon=self.config.epsilon, # Use getattr for backwards compat
            quant_config=None
        ).to(self.device)
        logging.info("Model structure initialized.")

        # --- Load Model Weights ---
        self._load_model_weights() # Load weights from self.checkpoint
        self.test_step = 0
        
        # --- GRPO Alpha Weights (only for FairPO testing) ---
        if not self.config.is_ref_training:
            self.alpha_privileged = self.checkpoint.get('alpha_privileged', None)
            self.alpha_non_privileged = self.checkpoint.get('alpha_non_privileged', None)

        # --- Setup WandB ---
        self._setup_wandb()

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

    def _setup_wandb(self):
        if self.config.is_wandb:
            wandb.init(
                project=self.config.wandb_project, # Use wandb_project here
                name=self.config.run_name,
                config=vars(self.config), # Log all argparse args
            )
            wandb.watch(self.model, log="all")
            
            # --- Define Custom Steps ---
            wandb.define_metric("test/step") # Master step for testing

            # --- Link metrics to steps ---
            # Test metrics use train/step
            wandb.define_metric("test/*", step_metric="test/step")
            logging.info(f"WandB initialized for project '{self.config.wandb_project}', run '{self.config.run_name}'.")

    def test(self):
        logging.info(f"Starting test...")
        self.model.eval()
        test_metrics = {
            'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
            'acc_priv': 0.0, 'acc_non_priv': 0.0, 'acc_overall': 0.0
        }
        processed_items = 0

        with torch.no_grad():
            test_iter = tqdm(self.test_loader, desc=f"Test", unit="batch", leave=False)
            for batch in test_iter:
                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)
                current_batch_size = pixels.size(0)

                output_dict = self.model(pixels=pixels, labels=labels)
                loss_components = output_dict['loss']
                acc_components = output_dict['acc']

                # Accumulate metrics (weighted sum by batch size)
                if self.config.is_ref_training:
                    loss = loss_components.get("loss", torch.tensor(np.nan, device=self.device))
                    if not torch.isnan(loss): test_metrics['loss_sft'] += loss.item() * current_batch_size
                else:
                    loss_priv = loss_components.get("privileged", torch.tensor(np.nan, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(np.nan, device=self.device))
                    test_metrics['loss_priv'] += loss_priv.item() * current_batch_size
                    test_metrics['loss_non_priv'] += loss_non_priv.item() * current_batch_size

                    # Use current training alphas for combined loss calculation (consistent reporting)
                    alpha_p = self.alpha_privileged
                    alpha_np = self.alpha_non_privileged
                    combined_loss = alpha_p * loss_priv + alpha_np * loss_non_priv
                    test_metrics['loss_total'] += combined_loss.item() * current_batch_size
                
                acc_priv_test = acc_components.get("privileged", np.nan)
                acc_non_priv_test = acc_components.get("non_privileged", np.nan)
                acc_overall_test = acc_components.get("acc", np.nan)

                test_metrics['acc_priv'] += acc_priv_test * current_batch_size
                test_metrics['acc_non_priv'] += acc_non_priv_test * current_batch_size
                test_metrics['acc_overall'] += acc_overall_test * current_batch_size

                processed_items += current_batch_size
                
                tqdm_postfix = {
                    'acc': acc_overall_test,
                    'acc_priv': acc_priv_test,
                    'acc_non_priv': acc_non_priv_test,
                }
                test_iter.set_postfix(**{k: f"{v:.3f}" if isinstance(v, float) else v for k, v in tqdm_postfix.items()})
    
        self.test_step += 1 # Increment test step counter *after* test loop
        
        # Calculate average test metrics and prefix keys
        avg_test_metrics = {f"test/{k}": v / processed_items for k, v in test_metrics.items()}

        logging.info(f"--- Test Results (Test Step {self.test_step}) ---")
        if self.config.is_ref_training:
            logging.info(f"  Avg Loss (SFT): {avg_test_metrics.get('test/loss_sft', np.nan):.4f}")
        else:
            logging.info(f"  Avg Loss (Priv): {avg_test_metrics.get('test/loss_priv', np.nan):.4f}")
            logging.info(f"  Avg Loss (Non-Priv): {avg_test_metrics.get('test/loss_non_priv', np.nan):.4f}")
            logging.info(f"  Avg Loss (Combined): {avg_test_metrics.get('test/loss_total', np.nan):.4f}") # Note: uses 'loss_total' key
        logging.info(f"  Avg Acc (Overall): {avg_test_metrics.get('test/acc_overall', np.nan):.4f}")
        logging.info(f"  Avg Acc (Priv): {avg_test_metrics.get('test/acc_priv', np.nan):.4f}")
        logging.info(f"  Avg Acc (Non-Priv): {avg_test_metrics.get('test/acc_non_priv', np.nan):.4f}")
        logging.info(f"---------------------------------------------")

        return avg_test_metrics

def main():
    parser = argparse.ArgumentParser(description="Test FairPO/SFT Model for Multi-Label Classification")

    # --- Determine User Dir (as before) ---
    if "raid" in str(Path.cwd()).lower(): user_dir = "/raid/speech/soumen"
    elif "home" in str(Path.cwd()).lower(): user_dir = "/home/soumen"
    else: user_dir = "."; logging.warning(f"Defaulting user_dir to '.'")
    default_coco_root = f"{user_dir}/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014"

    # --- Required Argument ---
    parser.add_argument('--checkpoint_path', type=str, default="output/ckpt/fairpo_model/FairPO-train/FairPO_ep5_lr5e-05_eta0.0001_eps0.1_beta2.0/checkpoint_epoch_2.pth", help='Path to the model checkpoint file to test')

    # --- Essential CLI Arguments for Test Setup ---
    parser.add_argument('--coco_root', type=str, default=default_coco_root, help='Root directory of the COCO dataset (may override checkpoint config if needed)')
    parser.add_argument('--index_dir', type=str, default=None, help='Directory for dataset index files (overrides checkpoint default if provided)')
    parser.add_argument('--test_frac', type=float, default=1.0, help='Fraction of test data (overrides checkpoint default)')
    parser.add_argument('--batch_size', type=int, default=32, help='Testing batch size (overrides checkpoint default)')

    # --- WandB Arguments (Control Test Logging) ---
    parser.add_argument('--wandb_project', type=str, default="FairPO-test", help='WandB project name for logging test results')
    parser.add_argument('--is_wandb', default=False, help='Enable WandB logging for this test run')

    args = parser.parse_args()

    tester = Tester(args)
    tester.test()

if __name__ == '__main__':
    main()