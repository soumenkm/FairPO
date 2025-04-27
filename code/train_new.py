import torch
import torch.optim as optim
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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Import necessary components from other files
from models import VisionModelForCLS 
# Use the OnDemand version for memory efficiency
from dataset import COCODatasetOnDemand, collate_fn_skip_none 

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
        logging.info(f"Using device: {self.device}")
        self.train_step = 0 # Initialize train step counter
        self.val_step = 0 # Initialize validation step counter

        # --- Checkpoint Setup FIRST (needed for wandb potentially) ---
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --- Load Data ---
        logging.info("Loading datasets...")
        
        # Define privileged indices based on config
        privileged_indices_set = set(map(int, config.privileged_indices.split(',')))
       
        self.train_dataset = COCODatasetOnDemand(
            root_dir=config.coco_root, frac=config.train_frac, is_train=True,
            privileged_indices_set=privileged_indices_set, seed=seed,
            index_dir=config.index_dir, force_regenerate=config.force_regenerate_index
        )
        self.val_dataset = COCODatasetOnDemand(
            root_dir=config.coco_root, frac=config.val_frac, is_train=False,
            privileged_indices_set=privileged_indices_set, seed=seed,
            index_dir=config.index_dir, force_regenerate=config.force_regenerate_index
        )

        self.label_names = self.train_dataset.get_label_names()
        self.num_labels = len(self.label_names)
        self.privileged_indices = self.train_dataset.privileged_indices
        self.non_privileged_indices = self.train_dataset.non_privileged_indices

        logging.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        logging.info(f"Num labels: {self.num_labels}")
        logging.info(f"Privileged Indices: {self.privileged_indices}")
        logging.info(f"Non-Privileged Indices: {self.non_privileged_indices}")

        # --- Create DataLoaders ---
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn_skip_none
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn_skip_none
        )

        # --- Initialize Model ---
        logging.info(f"Initializing model: {config.model_name}")
        ref_weights_path = Path(config.ref_cls_weights_path) if config.ref_cls_weights_path else None
        if not config.is_ref_training:
            if ref_weights_path is None or not ref_weights_path.is_file():
                logging.error(f"Reference weights not found at '{config.ref_cls_weights_path}'. Required for FairPO.")
                sys.exit(1)
            logging.info(f"Found reference classifier weights at: {ref_weights_path}")

        self.model = VisionModelForCLS(
            device=self.device, model_name=config.model_name, num_labels=self.num_labels,
            ref_cls_weights_path=str(ref_weights_path) if ref_weights_path else None,
            privileged_indices=self.privileged_indices, non_privileged_indices=self.non_privileged_indices,
            is_ref=config.is_ref_training, beta=config.beta, epsilon=config.epsilon,
            quant_config=None 
        ).to(self.device)

        self.model.calc_num_params()

        # --- Optimizer ---
        # Filter parameters that require gradients (only the classifier head)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            logging.error("No trainable parameters found in the model. Check model definition and requires_grad flags.")
            sys.exit(1)
        logging.info(f"Found {sum(p.numel() for p in trainable_params)} trainable parameters.")

        self.optimizer = optim.AdamW(
            trainable_params, # Pass only trainable parameters
            lr=config.learning_rate, weight_decay=config.weight_decay
        )

        # --- GRPO Alpha Weights (only for FairPO training) ---
        if not config.is_ref_training:
            self.alpha_privileged = torch.tensor(0.5, device=self.device)
            self.alpha_non_privileged = torch.tensor(0.5, device=self.device)

        # --- Checkpoint Setup (moved earlier) ---
        self.start_epoch = 0
        
        # --- Setup WandB ---
        self._setup_wandb() # Initializes wandb if enabled

    def _calculate_norm(self, model_params, norm_type=2.0):
        """Calculates the total norm for a list of parameters."""
        total_norm = torch.tensor(0.0, device=self.device)
        for p in model_params:
            param_norm = p.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def _calculate_grad_norm(self, model_params, norm_type=2.0):
        """Calculates the total norm for gradients of model parameters."""
        total_norm = torch.tensor(0.0, device=self.device)
        params_to_norm = [p for p in model_params if p.grad is not None]
        if not params_to_norm:
            return total_norm # Return 0 if no gradients
        for p in params_to_norm:
             param_norm = p.grad.data.norm(norm_type)
             total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def _setup_wandb(self):
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project, # Use wandb_project here
                name=self.config.run_name,
                config=vars(self.config), # Log all argparse args
            )
            wandb.watch(self.model, log="all")
            
            # --- Define Custom Steps ---
            wandb.define_metric("train/step") # Master step for training
            wandb.define_metric("val/step")   # Master step for validation

            # --- Link metrics to steps ---
            # Train metrics use train/step
            wandb.define_metric("train/*", step_metric="train/step")
            # Validation metrics use val/step
            wandb.define_metric("val/*", step_metric="val/step")
            logging.info(f"WandB initialized for project '{self.config.wandb_project}', run '{self.config.run_name}'.")

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if not self.config.is_ref_training:
            checkpoint_data['alpha_privileged'] = self.alpha_privileged
            checkpoint_data['alpha_non_privileged'] = self.alpha_non_privileged

        filename = f"checkpoint_epoch_{epoch+1}.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint_data, filepath)
        logging.info(f"Checkpoint saved to {filepath}")

        if is_best:
            best_filename = "checkpoint_best.pth"
            best_filepath = self.checkpoint_dir / best_filename
            torch.save(checkpoint_data, best_filepath)
            logging.info(f"Best checkpoint saved to {best_filepath}")

    def _load_checkpoint(self, checkpoint_path):
        pass # Implement if needed
    
    def train(self):
        logging.info("Starting training...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            epoch_metrics = {
                'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
                'acc_priv': 0.0, 'acc_non_priv': 0.0, 'acc_overall': 0.0,
                'grad_norm': 0.0, 'param_norm': 0.0 
            }
            processed_items = 0 # Count total items processed in the epoch
            batch_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", unit="batch")
            start_time = time.time()

            for i, batch in enumerate(batch_iter):
                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output_dict = self.model(pixels=pixels, labels=labels)
                loss_components = output_dict['loss']
                acc_components = output_dict['acc']
                current_batch_size = pixels.size(0)
                batch_log_payload = {} # For logging this specific batch

                # --- Loss Calculation & Backpropagation ---
                if self.config.is_ref_training:
                    # --- SFT Training ---
                    loss = loss_components.get("loss", torch.tensor(0.0, device=self.device))
                    batch_log_payload["train/sft_loss"] = loss.item()
                    epoch_metrics['loss_sft'] += loss.item() * current_batch_size # Weighted sum
                else:
                    # --- FairPO Training ---
                    loss_priv = loss_components.get("privileged", torch.tensor(0.0, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(0.0, device=self.device))

                    batch_log_payload["train/privileged_loss"] = loss_priv.item()
                    batch_log_payload["train/non_privileged_loss"] = loss_non_priv.item()
                    epoch_metrics['loss_priv'] += loss_priv.item() * current_batch_size # Weighted sum
                    epoch_metrics['loss_non_priv'] += loss_non_priv.item() * current_batch_size # Weighted sum

                    # GRPO Alpha Update
                    with torch.no_grad():
                        new_alpha_priv = self.alpha_privileged * torch.exp(self.config.eta_alpha * loss_priv)
                        new_alpha_non_priv = self.alpha_non_privileged * torch.exp(self.config.eta_alpha * loss_non_priv)
                        Z = new_alpha_priv + new_alpha_non_priv + 1e-8
                        self.alpha_privileged = new_alpha_priv / Z
                        self.alpha_non_privileged = new_alpha_non_priv / Z

                    batch_log_payload["train/alpha_p"] = self.alpha_privileged.item()
                    batch_log_payload["train/alpha_np"] = self.alpha_non_privileged.item()

                    # Combined loss for backpropagation
                    loss = self.alpha_privileged * loss_priv + self.alpha_non_privileged * loss_non_priv
                    batch_log_payload["train/combined_loss"] = loss.item()
                    epoch_metrics['loss_total'] += loss.item() * current_batch_size # Weighted sum

                # --- Backpropagate the determined loss ---
                loss.backward()

                # --- Calculate Gradient Norm (before clipping) ---
                grad_norm = self._calculate_grad_norm(trainable_params)
                batch_log_payload["train/grad_norm"] = grad_norm.item()
                epoch_metrics['grad_norm'] += grad_norm.item() # Simple sum for averaging later

                # --- Gradient Clipping ---
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.grad_clip)

                # --- Optimizer Step ---
                self.optimizer.step()
                self.train_step += 1 # Increment train step after successful step

                # --- Calculate Parameter Norm ---
                param_norm = self._calculate_norm(trainable_params)
                batch_log_payload["train/param_norm"] = param_norm.item()
                epoch_metrics['param_norm'] += param_norm.item() # Simple sum for averaging later
                
                # --- Accuracy Accumulation ---
                acc_priv_val = acc_components.get("privileged", np.nan)
                acc_non_priv_val = acc_components.get("non_privileged", np.nan)
                acc_overall_val = acc_components.get("acc", np.nan)

                if not np.isnan(acc_priv_val): epoch_metrics['acc_priv'] += acc_priv_val * current_batch_size
                if not np.isnan(acc_non_priv_val): epoch_metrics['acc_non_priv'] += acc_non_priv_val * current_batch_size
                if not np.isnan(acc_overall_val): epoch_metrics['acc_overall'] += acc_overall_val * current_batch_size

                batch_log_payload["train/accuracy_overall"] = acc_overall_val
                batch_log_payload["train/accuracy_privileged"] = acc_priv_val
                batch_log_payload["train/accuracy_non_privileged"] = acc_non_priv_val
                batch_log_payload["train/learning_rate"] = self.optimizer.param_groups[0]['lr']
                batch_log_payload["train/step"] = self.train_step # Add the step itself to log

                processed_items += current_batch_size # Use total items processed

                # --- Logging ---
                batch_log_payload["epoch"] = epoch + (i+1) / len(self.train_loader) # Fractional epoch
                if self.config.wandb_project:
                    wandb.log(batch_log_payload) # Log step metrics (uses train/step automatically)

                # Update tqdm progress bar
                tqdm_postfix = {
                    'loss': batch_log_payload.get('train/combined_loss', batch_log_payload.get('train/sft_loss', 0.0)),
                    'acc': batch_log_payload.get('train/accuracy_overall', 0.0),
                    'gnorm': batch_log_payload.get('train/grad_norm', 0.0),
                    'step': self.train_step
                }
                if not self.config.is_ref_training:
                    tqdm_postfix['alpha_p'] = batch_log_payload.get('train/alpha_p', 0.0)
                batch_iter.set_postfix(**{k: f"{v:.3f}" if isinstance(v, float) else v for k, v in tqdm_postfix.items()})

            # --- End of Epoch ---
            epoch_duration = time.time() - start_time
            logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")

            # --- Validation ---
            val_metrics = self._validate(epoch) # Returns dict with 'val/...' keys

            # Determine the primary validation metric for checkpointing
            if self.config.is_ref_training:
                current_val_metric = val_metrics.get('val/loss_sft', float('inf'))
            else:
                # Use combined loss for checkpointing FairPO
                current_val_metric = val_metrics.get('val/loss_total', float('inf')) # Note: loss_total used here

            # Log Validation Metrics (against val/step)
            val_metrics["val/step"] = self.val_step # Add the val step
            val_metrics["epoch"] = epoch + 1 # Also log epoch for reference
            wandb.log(val_metrics) # Log val metrics (uses val/step)

            # --- Checkpointing ---
            self._save_checkpoint(epoch, is_best=False)

        logging.info("Training finished.")
        if self.config.wandb_project:
            logging.info("Finishing WandB run.")
            wandb.finish()

    def _validate(self, epoch):
        logging.info(f"Starting validation for epoch {epoch+1}...")
        self.model.eval()
        val_metrics = {
            'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
            'acc_priv': 0.0, 'acc_non_priv': 0.0, 'acc_overall': 0.0
        }
        processed_items = 0

        with torch.no_grad():
            val_iter = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", unit="batch", leave=False)
            for batch in val_iter:
                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)
                current_batch_size = pixels.size(0)

                output_dict = self.model(pixels=pixels, labels=labels)
                loss_components = output_dict['loss']
                acc_components = output_dict['acc']

                # Accumulate metrics (weighted sum by batch size)
                if self.config.is_ref_training:
                    loss = loss_components.get("loss", torch.tensor(np.nan, device=self.device))
                    if not torch.isnan(loss): val_metrics['loss_sft'] += loss.item() * current_batch_size
                else:
                    loss_priv = loss_components.get("privileged", torch.tensor(np.nan, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(np.nan, device=self.device))
                    val_metrics['loss_priv'] += loss_priv.item() * current_batch_size
                    val_metrics['loss_non_priv'] += loss_non_priv.item() * current_batch_size

                    # Use current training alphas for combined loss calculation (consistent reporting)
                    alpha_p = self.alpha_privileged.item() if hasattr(self, 'alpha_privileged') else 0.5
                    alpha_np = self.alpha_non_privileged.item() if hasattr(self, 'alpha_non_privileged') else 0.5
                    combined_loss = alpha_p * loss_priv + alpha_np * loss_non_priv
                    val_metrics['loss_total'] += combined_loss.item() * current_batch_size
    
                acc_priv_val = acc_components.get("privileged", np.nan)
                acc_non_priv_val = acc_components.get("non_privileged", np.nan)
                acc_overall_val = acc_components.get("acc", np.nan)

                val_metrics['acc_priv'] += acc_priv_val * current_batch_size
                val_metrics['acc_non_priv'] += acc_non_priv_val * current_batch_size
                val_metrics['acc_overall'] += acc_overall_val * current_batch_size

                processed_items += current_batch_size

        self.val_step += 1 # Increment validation step counter *after* validation loop
        
        # Calculate average validation metrics and prefix keys
        avg_val_metrics = {f"val/{k}": v / processed_items for k, v in val_metrics.items()}

        logging.info(f"--- Validation Epoch {epoch+1} Results (Val Step {self.val_step}) ---")
        if self.config.is_ref_training:
            logging.info(f"  Avg Loss (SFT): {avg_val_metrics.get('val/loss_sft', np.nan):.4f}")
        else:
            logging.info(f"  Avg Loss (Priv): {avg_val_metrics.get('val/loss_priv', np.nan):.4f}")
            logging.info(f"  Avg Loss (Non-Priv): {avg_val_metrics.get('val/loss_non_priv', np.nan):.4f}")
            logging.info(f"  Avg Loss (Combined): {avg_val_metrics.get('val/loss_total', np.nan):.4f}") # Note: uses 'loss_total' key
        logging.info(f"  Avg Acc (Overall): {avg_val_metrics.get('val/acc_overall', np.nan):.4f}")
        logging.info(f"  Avg Acc (Priv): {avg_val_metrics.get('val/acc_priv', np.nan):.4f}")
        logging.info(f"  Avg Acc (Non-Priv): {avg_val_metrics.get('val/acc_non_priv', np.nan):.4f}")
        logging.info(f"---------------------------------------------")

        self.model.train() # Set back to training mode
        return avg_val_metrics

def main():
    parser = argparse.ArgumentParser(description="Train/Test FairPO Model for Multi-Label Classification")

    if "raid" in str(Path.cwd()):
        user_dir = "/raid/speech/soumen"
    else:
        user_dir = "home/soumen"
    current_dir = Path.cwd()
    
    # Paths
    parser.add_argument('--coco_root', type=str, default=f"{user_dir}/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014", help='Root directory of the COCO dataset') # CHANGE THIS
    parser.add_argument('--index_dir', type=str, default=None, help='Directory for dataset index files (default: coco_root/.index_cache)')
    parser.add_argument('--checkpoint_dir', type=str, default='./output/ckpt/fairpo_model', help='Directory to save checkpoints')
    parser.add_argument('--ref_cls_weights_path', type=str, default=f"{current_dir}/output/ckpt/ref_model/ckpt_ep_latest.pth", help='Path to SFT pre-trained reference classifier weights (REQUIRED for FairPO training/testing)') # Make default None

    # Model & Data
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224', help='Vision Transformer model name')
    parser.add_argument('--train_frac', type=float, default=1.0, help='Fraction of training data (0.0 to 1.0)')
    parser.add_argument('--val_frac', type=float, default=1.0, help='Fraction of validation data (0.0 to 1.0)')
    parser.add_argument('--privileged_indices', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19", help='Comma-separated privileged label indices')
    parser.add_argument('--force_regenerate_index', default=False, help='Force regeneration of dataset index')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Optimizer weight decay')
    parser.add_argument('--beta', type=float, default=2.0, help='DPO beta hyperparameter')
    parser.add_argument('--epsilon', type=float, default=0, help='Constraint slack epsilon')
    parser.add_argument('--eta_alpha', type=float, default=0.0001, help='Learning rate for GRPO alpha weights')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')

    # Training Mode
    parser.add_argument('--is_ref_training', default=False, help='Train the reference model (SFT) only.')

    # System
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--cpu', default=False, help='Force use CPU')

    # WandB
    parser.add_argument('--wandb_project', type=str, default="FairPO", help='WandB project name (disable if blank)')
    parser.add_argument('--run_name', type=str, default=None, help='Custom WandB run name (e.g., FairPO_b64_lr1e-5_eta0.01)')

    args = parser.parse_args()

    # Dynamic Run Name if not provided and WandB is enabled
    if args.wandb_project and not args.run_name:
        mode = "SFT" if args.is_ref_training else "FairPO"
        lr = args.learning_rate
        eta_str = f"_eta{args.eta_alpha}" if not args.is_ref_training else ""
        args.run_name = f"{mode}_bs{args.batch_size}_lr{lr}{eta_str}_{time.strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Generated WandB run name: {args.run_name}")
  
    # Set default index_dir if None
    if args.index_dir is None:
        args.index_dir = Path(args.coco_root) / '.index_cache'
        logging.info(f"Using default index directory: {args.index_dir}")
    args.index_dir = Path(args.index_dir) # Ensure it's a Path object
    args.index_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Trainer
    trainer = ModelTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()