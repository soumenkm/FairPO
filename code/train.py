# train.py
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
from typing import Dict, Any, Tuple, Optional # Added for type hinting

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" # Example default

# Import necessary components from other files
from models import VisionModelForCLS
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
    # --- __init__ remains the same as your provided version ---
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
        logging.info(f"Using device: {self.device}")
        self.train_step = 0
        self.val_step = 0
        self.start_epoch = 0 # Added for potential checkpoint loading logic later

        # Create checkpoint dir structure based on project/run name
        run_name = self._get_run_name() # Get run name early
        self.config.run_name = run_name # Store it back in config
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.wandb_project / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints will be saved in: {self.checkpoint_dir}")

        logging.info("Loading datasets...")
        privileged_indices_set = set(map(int, config.privileged_indices.split(','))) if config.privileged_indices else set()

        # Setup index dir path correctly
        index_dir_path = Path(config.index_dir) if config.index_dir else Path(config.coco_root) / '.index_cache'
        index_dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using index directory: {index_dir_path}")
        self.config.index_dir = index_dir_path # Update config with resolved path

        self.train_dataset = COCODatasetOnDemand(
            root_dir=config.coco_root, frac=config.train_frac, split_name="train",
            privileged_indices_set=privileged_indices_set, seed=seed,
            index_dir=self.config.index_dir, force_regenerate=config.force_regenerate_index
        )
        self.val_dataset = COCODatasetOnDemand(
            root_dir=config.coco_root, frac=config.val_frac, split_name="val",
            privileged_indices_set=privileged_indices_set, seed=seed,
            index_dir=self.config.index_dir, force_regenerate=config.force_regenerate_index
        )

        self.label_names = self.train_dataset.get_label_names()
        self.num_labels = len(self.label_names)
        self.privileged_indices = self.train_dataset.privileged_indices
        self.non_privileged_indices = self.train_dataset.non_privileged_indices

        logging.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        logging.info(f"Num labels: {self.num_labels}")
        logging.info(f"Privileged Indices: {self.privileged_indices}")
        logging.info(f"Non-Privileged Indices: {self.non_privileged_indices}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn_skip_none
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn_skip_none
        )

        logging.info(f"Initializing model: {config.model_name}")
        ref_weights_path = Path(config.ref_cls_weights_path) if config.ref_cls_weights_path else None
        # Check reference weights path ONLY IF not training the reference model
        if not config.is_ref_training:
            if ref_weights_path is None or not ref_weights_path.is_file():
                logging.error(f"Reference weights REQUIRED but not found at '{config.ref_cls_weights_path}' for FairPO training.")
                sys.exit(1)
            logging.info(f"Found reference classifier weights for FairPO: {ref_weights_path}")
        elif ref_weights_path and ref_weights_path.is_file():
             # If training ref model, but path is given, warn but proceed (don't load it)
             logging.warning(f"Reference weights path '{ref_weights_path}' provided, but ignored as --is_ref_training is True.")
             ref_weights_path = None # Ensure it's not passed to VisionModelForCLS

        # Pass loss_type from config to model __init__
        self.model = VisionModelForCLS(
            device=self.device, model_name=config.model_name, num_labels=self.num_labels,
            ref_cls_weights_path=str(ref_weights_path) if ref_weights_path else None,
            privileged_indices=self.privileged_indices, non_privileged_indices=self.non_privileged_indices,
            is_ref=config.is_ref_training,
            loss_type=getattr(config, 'loss_type', None), # Use getattr for safety if loss_type not in older configs
            beta=config.beta, epsilon=config.epsilon,
            quant_config=None
        ).to(self.device)

        self.model.calc_num_params()

        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable_params:
            logging.error("No trainable parameters found!")
            sys.exit(1)
        logging.info(f"Found {sum(p.numel() for p in self.trainable_params):,} trainable parameters.")

        self.optimizer = optim.AdamW(
            self.trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
        )

        if not config.is_ref_training:
            self.alpha_privileged = torch.tensor(0.5, device=self.device)
            self.alpha_non_privileged = torch.tensor(0.5, device=self.device)

        self._setup_wandb()

    # --- Norm Calculation Helpers (Unchanged) ---
    def _calculate_norm(self, model_params, norm_type=2.0):
        total_norm = torch.tensor(0.0, device=self.device)
        for p in model_params:
            if p is not None: # Added check for None params
                 param_norm = p.norm(norm_type)
                 total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def _calculate_grad_norm(self, model_params, norm_type=2.0):
        total_norm = torch.tensor(0.0, device=self.device)
        params_to_norm = [p for p in model_params if p is not None and p.grad is not None] # Added check
        if not params_to_norm: return total_norm
        for p in params_to_norm:
             param_norm = p.grad.data.norm(norm_type)
             total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    # --- WandB Setup Helper (Unchanged) ---
    def _setup_wandb(self):
        if self.config.is_wandb:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.run_name,
                    config=vars(self.config)
                )
                wandb.watch(self.model, log="all", log_freq=100) # Log gradients less frequently
                wandb.define_metric("train/step"); wandb.define_metric("val/step")
                wandb.define_metric("train/*", step_metric="train/step")
                wandb.define_metric("val/*", step_metric="val/step")
                wandb.define_metric("epoch", step_metric="train/step") # Link epoch to train step
                wandb.define_metric("epoch", step_metric="val/step")   # Link epoch to val step
                logging.info(f"WandB initialized for project '{self.config.wandb_project}', run '{self.config.run_name}'.")
            except Exception as e:
                 logging.error(f"Failed to initialize WandB: {e}")
                 self.config.is_wandb = False # Disable if init fails
        else:
            logging.info("WandB disabled.")

    # --- Checkpoint Helper ---
    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step, # Save steps
            'val_step': self.val_step,
            "config": vars(self.config) # Save config
        }
        if not self.config.is_ref_training:
            checkpoint_data['alpha_privileged'] = self.alpha_privileged
            checkpoint_data['alpha_non_privileged'] = self.alpha_non_privileged

        # Save epoch specific
        filename = f"checkpoint_epoch_{epoch+1}.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint_data, filepath)
        logging.info(f"Checkpoint saved: {filepath}")

        if is_best:
            best_filepath = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint_data, best_filepath)
            logging.info(f"Best checkpoint saved: {best_filepath}")

    # --- Helper to process a batch (forward pass + metric computation) ---
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Optional[Tuple[Dict[str, Any], int]]:
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
    def _update_epoch_metrics(self, epoch_metrics: Dict[str, Any], results_dict: Dict[str, Any], batch_size: int):
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

    # --- Helper for optimization step ---
    def _perform_optimization_step(self, loss: torch.Tensor) -> Optional[float]:
        """Performs backward pass, gradient clipping, and optimizer step."""
        loss.backward()
        grad_norm = self._calculate_grad_norm(self.trainable_params) # Before clipping

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.grad_clip)

        self.optimizer.step()
        return grad_norm.item() # Return scalar grad norm

    # --- Helper to calculate and log batch metrics (training only) ---
    def _log_wandb_train_batch_metrics(self, results_dict: Dict[str, Any], grad_norm: Optional[float], param_norm: float, epoch: int, batch_idx: int, num_batches: int):
        """Logs detailed metrics for a single training batch."""
        if not self.config.is_wandb: return # Skip if wandb disabled

        batch_log = {'train/step': self.train_step, 'epoch': epoch + (batch_idx + 1) / num_batches}

        # Losses
        loss_comp = results_dict.get('loss')
        if loss_comp:
            if self.config.is_ref_training:
                batch_log['train/sft_loss'] = loss_comp.get('loss', torch.tensor(np.nan)).item()
            else:
                batch_log['train/privileged_loss'] = loss_comp.get('privileged', torch.tensor(np.nan)).item()
                batch_log['train/non_privileged_loss'] = loss_comp.get('non_privileged', torch.tensor(np.nan)).item()
                batch_log['train/combined_loss'] = loss_comp.get('combined_loss', torch.tensor(np.nan)).item() 
                batch_log["train/alpha_p"] = self.alpha_privileged.item()
                batch_log["train/alpha_np"] = self.alpha_non_privileged.item()

        # Accuracies
        acc_comp = results_dict.get('acc')
        if acc_comp:
            batch_log['train/accuracy_overall'] = acc_comp.get('acc')
            batch_log['train/accuracy_privileged'] = acc_comp.get('privileged')
            batch_log['train/accuracy_non_privileged'] = acc_comp.get('non_privileged')
            
        em_comp = results_dict.get('em')
        if em_comp:
            batch_log['train/exact_match_overall'] = em_comp.get('em')
            batch_log['train/exact_match_privileged'] = em_comp.get('em_privileged')
            batch_log['train/exact_match_non_privileged'] = em_comp.get('em_non_privileged')

        # F1 Scores
        f1_comp = results_dict.get('f1')
        if f1_comp:
            batch_log['train/f1'] = f1_comp.get('macro_f1')
            batch_log['train/f1_privileged'] = f1_comp.get('f1_privileged')
            batch_log['train/f1_non_privileged'] = f1_comp.get('f1_non_privileged')

        # mAP Scores
        map_comp = results_dict.get('map')
        if map_comp:
            batch_log['train/map_overall'] = map_comp.get('mAP')
            batch_log['train/map_privileged'] = map_comp.get('mAP_privileged')
            batch_log['train/map_non_privileged'] = map_comp.get('mAP_non_privileged')

        # Norms and LR
        batch_log['train/grad_norm'] = grad_norm
        batch_log['train/param_norm'] = param_norm
        batch_log['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
        wandb.log({k: v for k, v in batch_log.items() if not isinstance(v, float) or not np.isnan(v)}) # Filter NaNs before logging

    # --- Helper to calculate epoch averages ---
    def _calculate_epoch_averages(self, epoch_metrics: Dict[str, Any], processed_items: int) -> Dict[str, float]:
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

    # --- Train Method ---
    def train(self):
        logging.info("Starting training...")
        best_val_metric = float('inf') # Or float('-inf') if maximizing

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            # Initialize accumulator dict with ALL expected metric keys
            epoch_metrics = {
                'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
                'acc_overall': 0.0, 'acc_priv': 0.0, 'acc_non_priv': 0.0,
                'em': 0.0, 'em_priv': 0.0, 'em_non_priv': 0.0,
                'f1': 0.0, 'f1_priv': 0.0, 'f1_non_priv': 0.0,
                'map_overall': 0.0, 'map_priv': 0.0, 'map_non_priv': 0.0
            }
            processed_items = 0
            processed_batches = 0
            num_batches = len(self.train_loader)

            batch_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", unit="batch")
            start_time = time.time()

            for i, batch in enumerate(batch_iter):
                # --- Process Batch (Forward + Metrics) ---
                processed_batch_info = self._process_batch(batch)
                results_dict, current_batch_size = processed_batch_info
                loss_components = results_dict['loss']
                self.optimizer.zero_grad()

                # --- Determine Loss for Backpropagation ---
                if self.config.is_ref_training:
                    loss = loss_components.get("loss", torch.tensor(0.0, device=self.device)) # SFT loss
                else: # FairPO training
                    loss_priv = loss_components.get("privileged", torch.tensor(0.0, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(0.0, device=self.device))

                    # GRPO Alpha Update (In-place update of self.alpha_...)
                    with torch.no_grad():
                        # Clamp loss before exp for stability
                        exp_arg_p = (self.config.eta_alpha * loss_priv).clamp(-10, 10)
                        exp_arg_np = (self.config.eta_alpha * loss_non_priv).clamp(-10, 10)
                        new_alpha_priv = self.alpha_privileged * torch.exp(exp_arg_p)
                        new_alpha_non_priv = self.alpha_non_privileged * torch.exp(exp_arg_np)
                        Z = new_alpha_priv + new_alpha_non_priv + 1e-8
                        self.alpha_privileged = new_alpha_priv / Z
                        self.alpha_non_privileged = new_alpha_non_priv / Z
            
                    # Combined loss for backpropagation
                    loss = self.alpha_privileged * loss_priv + self.alpha_non_privileged * loss_non_priv # Note: loss_components['loss'] is a placeholder sum, not used for backprop
                    results_dict['loss']['combined_loss'] = loss # Store combined loss for logging
                    
                # --- Optimization Step ---
                grad_norm_val = self._perform_optimization_step(loss)
        
                # --- Post-Optimization Steps ---
                self.train_step += 1
                param_norm_val = self._calculate_norm(self.trainable_params).item()

                # --- Accumulate Metrics ---
                self._update_epoch_metrics(epoch_metrics, results_dict, current_batch_size)
                processed_items += current_batch_size
                processed_batches += 1

                # --- Log Batch Metrics (conditional) ---
                if self.config.is_wandb:
                    self._log_wandb_train_batch_metrics(results_dict, grad_norm_val, param_norm_val, epoch, i, num_batches)

                # --- Update TQDM ---
                tqdm_postfix = {
                    'loss': loss.item(), # Show loss used for backprop
                    'acc': results_dict.get('acc', {}).get('acc'),
                    'step': self.train_step
                }
                if not self.config.is_ref_training:
                    tqdm_postfix['alpha_p'] = self.alpha_privileged.item()
                batch_iter.set_postfix(**{k: f"{v:.3f}" if isinstance(v, float) else v for k, v in tqdm_postfix.items()})

            # --- End of Epoch ---
            epoch_duration = time.time() - start_time
            logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")

            # Calculate average epoch metrics
            avg_train_metrics = self._calculate_epoch_averages(epoch_metrics, processed_items)
            # Prefix keys with 'train/'
            avg_train_metrics = {f"train/{k.replace('avg_', '')}": v for k, v in avg_train_metrics.items()}

            # Validation
            avg_val_metrics = self._validate(epoch) # Returns dict with 'val/...' prefixed keys

            # Log validation summary
            self._log_wandb_validation_summary(epoch, avg_val_metrics)

            # Determine best metric for checkpointing
            # Use validation loss as the default metric to minimize
            if self.config.is_ref_training:
                current_val_metric = avg_val_metrics.get('val/loss_sft')
            else:
                # For FairPO, maybe use combined loss, worst-group metric, or overall performance?
                # Using combined loss for now as per previous logic
                current_val_metric = avg_val_metrics.get('val/loss_total')

            is_best = current_val_metric < best_val_metric
            if is_best:
                best_val_metric = current_val_metric
                logging.info(f"New best validation metric: {best_val_metric:.4f}")

            # Save checkpoint
            self._save_checkpoint(epoch, is_best=is_best)
            
            # Log epoch summary to console
            self._log_epoch_summary_console(epoch, avg_train_metrics, mode='train')
            self._log_epoch_summary_console(epoch, avg_val_metrics, mode='val')

        logging.info("Training finished.")
        if self.config.is_wandb:
            logging.info("Finishing WandB run.")
            wandb.finish()

    # --- Validate Method ---
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Runs validation loop and returns averaged metrics."""
        logging.info(f"Starting validation for epoch {epoch+1}...")
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
            val_iter = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", unit="batch", leave=False)
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
        avg_val_metrics = {f"val/{k.replace('avg_', '')}": v for k, v in avg_val_metrics_raw.items()}

        # Increment val_step AFTER calculations for this epoch
        self.val_step += 1

        # Log validation summary to console
        self._log_wandb_validation_summary(epoch, avg_val_metrics)

        self.model.train() # Set back to training mode
        return avg_val_metrics # Return dict with 'val/' prefixes

    # --- Helper to log validation summary (console) ---
    def _log_epoch_summary_console(self, epoch: int, avg_epoch_metrics: Dict[str, float], mode: str):
        """Logs the summary metrics for an epoch (train or val) to the console."""
        if mode not in ['train', 'val']:
            logging.warning(f"Invalid mode '{mode}' provided to _log_epoch_summary_console. Skipping log.")
            return

        # Determine prefix based on mode
        prefix = f"{mode}/"
        title_prefix = "Training" if mode == 'train' else "Validation"
        step_info = f"(Train Step {self.train_step})" if mode == 'train' else f"(Val Step {self.val_step})"

        logging.info(f"--- {title_prefix} Epoch {epoch+1} Results {step_info} ---")

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

    # --- Helper to log epoch summary (WandB) ---
    def _log_wandb_validation_summary(self, epoch: int, val_metrics: Dict[str, float]):
        """Logs average training and validation metrics for the epoch to WandB."""
        if not self.config.is_wandb: return

        log_payload = {}
        log_payload.update(val_metrics)

        # Add step info
        log_payload['train/step'] = self.train_step
        log_payload['val/step'] = self.val_step
        log_payload['epoch'] = epoch + 1 # Log integer epoch

        wandb.log({k: v for k, v in log_payload.items() if not isinstance(v, float) or not np.isnan(v)}) # Filter NaNs
        
    # --- Helper to generate run name ---
    def _get_run_name(self) -> str:
        """Generates a descriptive run name based on config."""
        if self.config.run_name: # Use provided name if exists
             return self.config.run_name

        # Generate dynamically
        mode = "SFT" if self.config.is_ref_training else "FairPO"
        loss_part = f"_{self.config.loss_type}" if not self.config.is_ref_training and self.config.loss_type else ""
        lr_str = f"lr{self.config.learning_rate:.0e}" # Scientific notation
        ep_str = f"ep{self.config.epochs}"
        frac_str = f"frac{self.config.train_frac:.2f}"

        # Add specific FairPO params if not SFT
        fairpo_params = ""
        if not self.config.is_ref_training:
            eta_str = f"eta{self.config.eta_alpha}"
            eps_str = f"_eps{self.config.epsilon}"
            beta_str = f"_beta{self.config.beta}"
            fairpo_params = f"{eta_str}{eps_str}{beta_str}_"

        run_name = f"{mode}{loss_part}_{lr_str}_{fairpo_params}{frac_str}_{ep_str}"
        logging.info(f"Generated run name: {run_name}")
        return run_name

# --- main function remains the same as your provided version ---
def main():
    parser = argparse.ArgumentParser(description="Train FairPO Model for Multi-Label Classification")

    # Detect user/system path
    if "raid" in str(Path.cwd()).lower(): user_dir = "/raid/speech/soumen"
    elif "home" in str(Path.cwd()).lower(): user_dir = "/home/soumen"
    else: user_dir = "."; logging.warning(f"Defaulting user_dir to '.'")
    current_dir = Path.cwd()
    default_coco_root = f"{user_dir}/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014"
 
    # Paths
    ref_cls_weights_path = f"{current_dir}/output/ckpt/FairPO-train/SFT_lr5e-05_frac1.00_ep5/checkpoint_best.pth" # !IMP: CHANGE THIS
    parser.add_argument('--coco_root', type=str, default=default_coco_root, help='Root directory of the COCO dataset') 
    parser.add_argument('--index_dir', type=str, default=None, help='Directory for dataset index files (default: coco_root/.index_cache)')
    parser.add_argument('--checkpoint_dir', type=str, default='./output/ckpt', help='Directory to save checkpoints')
    parser.add_argument('--ref_cls_weights_path', type=str, default=ref_cls_weights_path, help='Path to SFT pre-trained reference classifier weights (REQUIRED for FairPO training/testing)') # Make default None

    # Model & Data
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224', help='Vision Transformer model name')
    parser.add_argument('--train_frac', type=float, default=1.0, help='Fraction of training data (0.0 to 1.0)') # !IMP: CHANGE THIS
    parser.add_argument('--val_frac', type=float, default=1.0, help='Fraction of validation data (0.0 to 1.0)') # !IMP: CHANGE THIS
    parser.add_argument('--privileged_indices', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19", help='Comma-separated privileged label indices')
    parser.add_argument('--force_regenerate_index', default=False, help='Force regeneration of dataset index')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs') 
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Optimizer learning rate') 
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Optimizer weight decay')
    parser.add_argument('--beta', type=float, default=2.0, help='DPO beta hyperparameter') 
    parser.add_argument('--epsilon', type=float, default=0.01, help='Constraint slack epsilon')
    parser.add_argument('--eta_alpha', type=float, default=0.0001, help='Learning rate for GRPO alpha weights')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')

    # Training Mode
    parser.add_argument('--is_ref_training', default=False, help='Train the reference model (SFT) only.') # !IMP: CHANGE THIS
    parser.add_argument('--loss_type', default='dpo', help='Loss type (dpo, simpo, cpo)') # If ref_training, this is ignored

    # System
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--cpu', default=False, help='Force use CPU')

    # WandB
    parser.add_argument('--wandb_project', type=str, default="FairPO-train", help='WandB project name (disable if blank)')
    parser.add_argument('--is_wandb', type=bool, default=True, help='Whether to use WandB for logging') # !IMP: CHANGE THIS
    parser.add_argument('--run_name', type=str, default=None, help='Custom WandB run name (e.g., FairPO_b64_lr1e-5_eta0.01)')

    args = parser.parse_args()

    # Initialize Trainer and start training
    trainer = ModelTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()