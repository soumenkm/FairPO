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

# Comment out or remove if not needed
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import necessary components from other files
from archive.models import VisionModelForCLS # Assuming models.py is in the same directory
# Use the OnDemand version for memory efficiency
from dataset import COCODatasetOnDemand, collate_fn_skip_none # Assuming dataset.py is in the same directory

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # Potentially make CuDNN deterministic, but can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
        logging.info(f"Using device: {self.device}")
        self.global_step = 0 # Initialize global step counter

        # --- Setup WandB ---
        self._setup_wandb()

        # --- Load Data ---
        logging.info("Loading datasets...")
        # Define privileged indices based on config
        all_labels = list(range(80)) # Assuming 80 COCO classes
        if config.privileged_indices:
             try:
                 privileged_indices_set = set(map(int, config.privileged_indices.split(',')))
                 if not all(0 <= idx < 80 for idx in privileged_indices_set):
                     raise ValueError("Privileged indices must be between 0 and 79.")
             except ValueError as e:
                 logging.error(f"Invalid privileged indices format or value: {config.privileged_indices}. Error: {e}")
                 sys.exit(1)
        else:
             privileged_indices_set = set(range(20)) # Default: first 20 classes
             logging.warning(f"No privileged indices specified. Defaulting to: {sorted(list(privileged_indices_set))}")

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
        self.test_dataset = COCODatasetOnDemand(
             root_dir=config.coco_root, frac=config.val_frac, is_train=False,
             privileged_indices_set=privileged_indices_set, seed=seed,
             index_dir=config.index_dir, force_regenerate=config.force_regenerate_index
        )

        self.label_names = self.train_dataset.get_label_names()
        self.num_labels = len(self.label_names)
        self.privileged_indices = self.train_dataset.privileged_indices
        self.non_privileged_indices = self.train_dataset.non_privileged_indices

        logging.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}, Test samples: {len(self.test_dataset)}")
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
        self.test_loader = DataLoader(
             self.test_dataset, batch_size=config.batch_size, shuffle=False,
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
            quant_config=None # Add quantization config if needed
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

        # --- Checkpoint Setup ---
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_metric = float('inf')
        self.start_epoch = 0

        # --- Load Checkpoint if specified ---
        if config.resume_checkpoint:
            self._load_checkpoint(config.resume_checkpoint)
            # Adjust global step based on loaded epoch and estimated steps per epoch
            self.global_step = self.start_epoch * len(self.train_loader)


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
        for p in model_params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm


    def _setup_wandb(self):
        if self.config.wandb_project:
            try:
                # Ensure checkpoint dir exists before wandb init if saving code
                self.checkpoint_dir = Path(self.config.checkpoint_dir)
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                wandb.init(project=self.config.project_name, name=self.config.run_name, config=self.config)
                wandb.watch(self.model, log="all")
                wandb.define_metric("train/step")
                wandb.define_metric("val/step")
                wandb.define_metric("train/*", step_metric="train/step")
                wandb.define_metric("val/*", step_metric="val/step")
            except Exception as e:
                 logging.error(f"Failed to initialize WandB: {e}")
                 self.config.wandb_project = None
        else:
            logging.info("WandB logging is disabled.")

    def _save_checkpoint(self, epoch, is_best=False):
        # Save only trainable parameters (classifier head)
        classifier_state_dict = self.model.model.classifier.state_dict()

        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': classifier_state_dict, # Save only classifier weights
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'global_step': self.global_step
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

            # Save SFT weights separately if training reference model and it's the best
            if self.config.is_ref_training:
                 sft_filename = "sft_classifier_weights_best.pth"
                 sft_filepath = self.checkpoint_dir / sft_filename
                 torch.save(classifier_state_dict, sft_filepath) # Save only the state dict
                 logging.info(f"Best SFT classifier weights saved to {sft_filepath}")
                 # Update config path ONLY if not originally provided, else respect user input
                 if self.config.ref_cls_weights_path is None:
                      self.config.ref_cls_weights_path = str(sft_filepath)


    def _load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load classifier weights carefully
            try:
                self.model.model.classifier.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                 logging.error(f"Error loading classifier state_dict: {e}")
                 logging.warning("Attempting to load with strict=False")
                 self.model.model.classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)


            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer state to the correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            except Exception as e:
                logging.warning(f"Could not load optimizer state: {e}. Optimizer will start from scratch.")


            self.start_epoch = checkpoint.get('epoch', 0) # Use get for backward compat.
            self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
            self.global_step = checkpoint.get('global_step', self.start_epoch * len(self.train_loader)) # Resume global step


            if not self.config.is_ref_training and 'alpha_privileged' in checkpoint:
                 self.alpha_privileged = checkpoint['alpha_privileged'].to(self.device)
                 self.alpha_non_privileged = checkpoint['alpha_non_privileged'].to(self.device)
                 logging.info(f"Loaded alpha weights: Priv={self.alpha_privileged.item():.4f}, NonPriv={self.alpha_non_privileged.item():.4f}")

            logging.info(f"Resumed training from epoch {self.start_epoch}, global step {self.global_step}")
        else:
            logging.warning(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")


    def train(self):
        logging.info("Starting training...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            epoch_metrics = {
                'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
                'acc_priv': 0.0, 'acc_non_priv': 0.0, 'acc_overall': 0.0,
                'grad_norm': 0.0, 'param_norm': 0.0 # Add accumulators for norms
            }
            processed_batches = 0
            batch_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", unit="batch")
            start_time = time.time()

            for i, batch in enumerate(batch_iter):
                if not batch or 'pixels' not in batch or batch['pixels'].numel() == 0:
                     logging.warning(f"Skipping empty batch {i+1} in epoch {epoch+1}.")
                     continue

                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output_dict = self.model(pixels=pixels, labels=labels)

                if output_dict is None or 'loss' not in output_dict or output_dict['loss'] is None:
                     logging.warning(f"Skipping batch {i+1} due to model output issue.")
                     continue

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

                    # Check for NaN before alpha update
                    if torch.isnan(loss_priv) or torch.isnan(loss_non_priv):
                        logging.warning(f"NaN loss detected (P:{loss_priv.item()}, NP:{loss_non_priv.item()}) at step {self.global_step}. Skipping batch.")
                        continue

                    # GRPO Alpha Update
                    with torch.no_grad():
                         # Use item() for exp to avoid tensor creation if losses are scalar
                         # Using self.config.eta_alpha here
                         new_alpha_priv = self.alpha_privileged * torch.exp(self.config.eta_alpha * loss_priv)
                         new_alpha_non_priv = self.alpha_non_privileged * torch.exp(self.config.eta_alpha * loss_non_priv)
                         Z = new_alpha_priv + new_alpha_non_priv + 1e-8
                         self.alpha_privileged = new_alpha_priv / Z
                         self.alpha_non_privileged = new_alpha_non_priv / Z

                    batch_log_payload["train/alpha_p"] = self.alpha_privileged.item()
                    batch_log_payload["train/alpha_np"] = self.alpha_non_privileged.item()

                    # Add non-priv loss AFTER alpha update (as its contribution depends on alpha)
                    epoch_metrics['loss_non_priv'] += loss_non_priv.item() * current_batch_size # Weighted sum

                    # Combined loss for backpropagation
                    loss = self.alpha_privileged * loss_priv + self.alpha_non_privileged * loss_non_priv
                    batch_log_payload["train/combined_loss"] = loss.item()
                    epoch_metrics['loss_total'] += loss.item() * current_batch_size # Weighted sum


                # --- Backpropagate the determined loss ---
                if torch.isnan(loss):
                    logging.warning(f"NaN loss detected before backward() at step {self.global_step}. Skipping batch.")
                    continue
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
                self.global_step += 1 # Increment global step

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

                processed_batches += current_batch_size # Use total items processed

                # --- Logging ---
                if self.global_step % self.config.log_freq == 0:
                    batch_log_payload["epoch"] = epoch + (i+1) / len(self.train_loader) # Fractional epoch
                    if self.config.wandb_project:
                        wandb.log(batch_log_payload, step=self.global_step)

                    # Update tqdm progress bar
                    tqdm_postfix = {
                        'loss': batch_log_payload.get('train/combined_loss', batch_log_payload.get('train/sft_loss', 0.0)),
                        'acc': batch_log_payload.get('train/accuracy_overall', 0.0),
                        'gnorm': batch_log_payload.get('train/grad_norm', 0.0),
                    }
                    if not self.config.is_ref_training:
                         tqdm_postfix['alpha_p'] = batch_log_payload.get('train/alpha_p', 0.0)
                    batch_iter.set_postfix(**{k: f"{v:.3f}" for k, v in tqdm_postfix.items()})


            # --- End of Epoch ---
            epoch_duration = time.time() - start_time
            logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")

            if processed_batches == 0:
                 logging.warning(f"Epoch {epoch+1} finished without processing any items. Check data.")
                 continue

            # Calculate average epoch metrics
            avg_metrics = {k: v / processed_batches for k, v in epoch_metrics.items() if k.startswith('loss') or k.startswith('acc')}
            # For norms, simple average over batches where it was computed
            num_norm_steps = len(self.train_loader) # Approx number of norm calculations
            avg_metrics['grad_norm'] = epoch_metrics['grad_norm'] / num_norm_steps if num_norm_steps > 0 else 0
            avg_metrics['param_norm'] = epoch_metrics['param_norm'] / num_norm_steps if num_norm_steps > 0 else 0


            # --- Validation ---
            val_metrics = self._validate(epoch)
            # Determine the primary validation metric for checkpointing
            if self.config.is_ref_training:
                current_val_metric = val_metrics.get('val/loss_sft', float('inf'))
            else:
                 # Use combined loss, or maybe overall accuracy? Let's stick to combined loss.
                 current_val_metric = val_metrics.get('val/loss_combined', float('inf'))


            # --- Log Epoch Metrics ---
            epoch_log_payload = {
                 "epoch": epoch + 1,
                 "train/epoch_duration_sec": epoch_duration,
                 "train/avg_accuracy_overall": avg_metrics.get('acc_overall', np.nan),
                 "train/avg_accuracy_privileged": avg_metrics.get('acc_priv', np.nan),
                 "train/avg_accuracy_non_privileged": avg_metrics.get('acc_non_priv', np.nan),
                 "train/avg_grad_norm": avg_metrics.get('grad_norm', np.nan),
                 "train/avg_param_norm": avg_metrics.get('param_norm', np.nan),
            }
            if self.config.is_ref_training:
                 epoch_log_payload["train/avg_sft_loss"] = avg_metrics.get('loss_sft', np.nan)
            else:
                 epoch_log_payload.update({
                     "train/avg_privileged_loss": avg_metrics.get('loss_priv', np.nan),
                     "train/avg_non_privileged_loss": avg_metrics.get('loss_non_priv', np.nan),
                     "train/avg_combined_loss": avg_metrics.get('loss_total', np.nan),
                     "train/final_alpha_p": self.alpha_privileged.item(),
                     "train/final_alpha_np": self.alpha_non_privileged.item(),
                 })
            # Add validation metrics
            epoch_log_payload.update(val_metrics)

            if self.config.wandb_project:
                wandb.log(epoch_log_payload, step=self.global_step) # Log at the end of the epoch step

            # --- Checkpointing ---
            is_best = current_val_metric < self.best_val_metric
            if is_best:
                self.best_val_metric = current_val_metric
                logging.info(f"New best validation metric: {self.best_val_metric:.4f} at epoch {epoch+1}")
            self._save_checkpoint(epoch, is_best=is_best)

        logging.info("Training finished.")
        if self.config.wandb_project and wandb.run is not None:
            # Log final best metric if run is still active
            wandb.summary['best_validation_metric'] = self.best_val_metric
            wandb.finish()


    def _validate(self, epoch):
        logging.info(f"Starting validation for epoch {epoch+1}...")
        self.model.eval()
        val_metrics = {
            'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
            'acc_priv': 0.0, 'acc_non_priv': 0.0, 'acc_overall': 0.0
        }
        processed_batches = 0

        with torch.no_grad():
            val_iter = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", unit="batch", leave=False)
            for batch in val_iter:
                if not batch or 'pixels' not in batch or batch['pixels'].numel() == 0:
                     continue

                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)
                current_batch_size = pixels.size(0)

                output_dict = self.model(pixels=pixels, labels=labels)

                if output_dict is None or 'loss' not in output_dict or output_dict['loss'] is None:
                     continue

                loss_components = output_dict['loss']
                acc_components = output_dict['acc']

                # Accumulate metrics (weighted sum by batch size)
                if self.config.is_ref_training:
                    loss = loss_components.get("loss", torch.tensor(np.nan, device=self.device))
                    if not torch.isnan(loss): val_metrics['loss_sft'] += loss.item() * current_batch_size
                else:
                    loss_priv = loss_components.get("privileged", torch.tensor(np.nan, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(np.nan, device=self.device))
                    if not torch.isnan(loss_priv): val_metrics['loss_priv'] += loss_priv.item() * current_batch_size
                    if not torch.isnan(loss_non_priv): val_metrics['loss_non_priv'] += loss_non_priv.item() * current_batch_size

                    # Use current training alphas for combined loss calculation (consistent reporting)
                    # Avoid NaN propagation if individual losses are NaN
                    if not torch.isnan(loss_priv) and not torch.isnan(loss_non_priv):
                        combined_loss = self.alpha_privileged * loss_priv + self.alpha_non_privileged * loss_non_priv
                        val_metrics['loss_total'] += combined_loss.item() * current_batch_size
                    else:
                         # If either is NaN, combined loss cannot be computed reliably
                         pass # Or accumulate NaN count if needed


                acc_priv_val = acc_components.get("privileged", np.nan)
                acc_non_priv_val = acc_components.get("non_privileged", np.nan)
                acc_overall_val = acc_components.get("acc", np.nan)

                if not np.isnan(acc_priv_val): val_metrics['acc_priv'] += acc_priv_val * current_batch_size
                if not np.isnan(acc_non_priv_val): val_metrics['acc_non_priv'] += acc_non_priv_val * current_batch_size
                if not np.isnan(acc_overall_val): val_metrics['acc_overall'] += acc_overall_val * current_batch_size

                processed_batches += current_batch_size

        if processed_batches == 0:
             logging.warning("Validation finished without processing any items.")
             return {} # Return empty dict

        # Calculate average validation metrics
        avg_val_metrics = {f"val/{k}": v / processed_batches for k, v in val_metrics.items()}


        logging.info(f"--- Validation Epoch {epoch+1} Results ---")
        if self.config.is_ref_training:
             logging.info(f"  Avg Loss (SFT): {avg_val_metrics.get('val/loss_sft', np.nan):.4f}")
        else:
             logging.info(f"  Avg Loss (Priv): {avg_val_metrics.get('val/loss_priv', np.nan):.4f}")
             logging.info(f"  Avg Loss (Non-Priv): {avg_val_metrics.get('val/loss_non_priv', np.nan):.4f}")
             logging.info(f"  Avg Loss (Combined): {avg_val_metrics.get('val/loss_total', np.nan):.4f}")
        logging.info(f"  Avg Acc (Overall): {avg_val_metrics.get('val/acc_overall', np.nan):.4f}")
        logging.info(f"  Avg Acc (Priv): {avg_val_metrics.get('val/acc_priv', np.nan):.4f}")
        logging.info(f"  Avg Acc (Non-Priv): {avg_val_metrics.get('val/acc_non_priv', np.nan):.4f}")
        logging.info(f"------------------------------------")

        self.model.train() # Set back to training mode
        return avg_val_metrics


    def test(self, checkpoint_path=None):
        logging.info("Starting testing...")
        # Load best checkpoint by default if not specified
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_best.pth"

        # Ensure reference weights are loaded for FairPO testing if needed
        ref_weights_loaded = False
        if not self.config.is_ref_training:
             if not hasattr(self.model.model, 'ref_classifier'):
                 logging.error("Model does not have a reference classifier, cannot perform FairPO test.")
                 return {}
             ref_weights_file = Path(self.config.ref_cls_weights_path) if self.config.ref_cls_weights_path else None
             if ref_weights_file and ref_weights_file.is_file():
                  try:
                      self.model.model.ref_classifier.load_state_dict(torch.load(ref_weights_file, map_location=self.device))
                      logging.info(f"Loaded reference weights from {ref_weights_file} for test.")
                      ref_weights_loaded = True
                  except Exception as e:
                      logging.error(f"Failed to load reference weights from {ref_weights_file}: {e}")
             if not ref_weights_loaded:
                  logging.warning("Reference weights needed for FairPO test NOT loaded. Test results might be inaccurate.")


        # Load the main classifier weights from the specified checkpoint
        if checkpoint_path.is_file():
             logging.info(f"Loading model checkpoint for testing: {checkpoint_path}")
             checkpoint = torch.load(checkpoint_path, map_location=self.device)
             if 'model_state_dict' in checkpoint:
                 try:
                    self.model.model.classifier.load_state_dict(checkpoint['model_state_dict'])
                    logging.info("Loaded classifier weights from checkpoint.")
                 except RuntimeError as e:
                    logging.error(f"Error loading test checkpoint classifier state_dict: {e}")
                    logging.warning("Attempting to load with strict=False")
                    self.model.model.classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)

             else: # Handle case where checkpoint is just the state dict
                 try:
                    self.model.model.classifier.load_state_dict(checkpoint)
                    logging.info("Loaded classifier weights directly from state dict file.")
                 except Exception as e:
                    logging.error(f"Failed to load state dict file {checkpoint_path}: {e}")
                    return {}
        else:
            logging.error(f"Test checkpoint not found at {checkpoint_path}. Cannot perform test.")
            return {}


        self.model.eval()
        test_metrics = {
            'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
            'acc_priv': 0.0, 'acc_non_priv': 0.0, 'acc_overall': 0.0
        }
        processed_batches = 0

        with torch.no_grad():
            test_iter = tqdm(self.test_loader, desc="Testing", unit="batch", leave=False)
            for batch in test_iter:
                if not batch or 'pixels' not in batch or batch['pixels'].numel() == 0:
                     continue
                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)
                current_batch_size = pixels.size(0)

                output_dict = self.model(pixels=pixels, labels=labels)

                if output_dict is None or 'loss' not in output_dict or output_dict['loss'] is None:
                     continue

                loss_components = output_dict['loss']
                acc_components = output_dict['acc']

                 # Accumulate metrics (similar to validation, weighted sum)
                if self.config.is_ref_training:
                    loss = loss_components.get("loss", torch.tensor(np.nan, device=self.device))
                    if not torch.isnan(loss): test_metrics['loss_sft'] += loss.item() * current_batch_size
                else:
                    loss_priv = loss_components.get("privileged", torch.tensor(np.nan, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(np.nan, device=self.device))
                    if not torch.isnan(loss_priv): test_metrics['loss_priv'] += loss_priv.item() * current_batch_size
                    if not torch.isnan(loss_non_priv): test_metrics['loss_non_priv'] += loss_non_priv.item() * current_batch_size

                    # Use final training alphas if available, else default 0.5
                    alpha_p = self.alpha_privileged.item() if hasattr(self, 'alpha_privileged') else 0.5
                    alpha_np = self.alpha_non_privileged.item() if hasattr(self, 'alpha_non_privileged') else 0.5

                    if not torch.isnan(loss_priv) and not torch.isnan(loss_non_priv):
                        combined_loss = alpha_p * loss_priv + alpha_np * loss_non_priv
                        test_metrics['loss_total'] += combined_loss.item() * current_batch_size
                    else:
                         pass

                acc_priv_val = acc_components.get("privileged", np.nan)
                acc_non_priv_val = acc_components.get("non_privileged", np.nan)
                acc_overall_val = acc_components.get("acc", np.nan)

                if not np.isnan(acc_priv_val): test_metrics['acc_priv'] += acc_priv_val * current_batch_size
                if not np.isnan(acc_non_priv_val): test_metrics['acc_non_priv'] += acc_non_priv_val * current_batch_size
                if not np.isnan(acc_overall_val): test_metrics['acc_overall'] += acc_overall_val * current_batch_size

                processed_batches += current_batch_size

        if processed_batches == 0:
            logging.error("Testing finished without processing any items.")
            return {}

        # Calculate average test metrics
        avg_test_metrics = {f"test/{k}": v / processed_batches for k, v in test_metrics.items()}

        logging.info("--- Test Results ---")
        if self.config.is_ref_training:
             logging.info(f"  Avg Loss (SFT): {avg_test_metrics.get('test/loss_sft', np.nan):.4f}")
        else:
             logging.info(f"  Avg Loss (Priv): {avg_test_metrics.get('test/loss_priv', np.nan):.4f}")
             logging.info(f"  Avg Loss (Non-Priv): {avg_test_metrics.get('test/loss_non_priv', np.nan):.4f}")
             logging.info(f"  Avg Loss (Combined): {avg_test_metrics.get('test/loss_total', np.nan):.4f}")
        logging.info(f"  Avg Acc (Overall): {avg_test_metrics.get('test/acc_overall', np.nan):.4f}")
        logging.info(f"  Avg Acc (Priv): {avg_test_metrics.get('test/acc_priv', np.nan):.4f}")
        logging.info(f"  Avg Acc (Non-Priv): {avg_test_metrics.get('test/acc_non_priv', np.nan):.4f}")
        logging.info("--------------------")

        # Log test results to WandB summary if enabled and run active
        if self.config.wandb_project and wandb.run is not None:
            for key, value in avg_test_metrics.items():
                 wandb.summary[key.replace('/', '_')] = value # Use underscore for summary keys
            logging.info("Test results logged to WandB summary.")

        return avg_test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train/Test FairPO Model for Multi-Label Classification")

    # Paths
    parser.add_argument('--coco_root', type=str, default="/home/soumen/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014", help='Root directory of the COCO dataset')
    parser.add_argument('--index_dir', type=str, default=None, help='Directory for dataset index files (default: coco_root/.index_cache)')
    parser.add_argument('--checkpoint_dir', type=str, default='./output/ckpt/fairpo_model', help='Directory to save checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--ref_cls_weights_path', type=str, default="/home/soumen/OML/FairPO/output/ckpt/ref_model/ckpt_ep_latest.pth", help='Path to SFT pre-trained reference classifier weights (REQUIRED for FairPO training/testing)')

    # Model & Data
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224', help='Vision Transformer model name')
    parser.add_argument('--train_frac', type=float, default=1.0, help='Fraction of training data (0.0 to 1.0)')
    parser.add_argument('--val_frac', type=float, default=1.0, help='Fraction of validation data (0.0 to 1.0)')
    parser.add_argument('--privileged_indices', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19", help='Comma-separated privileged label indices')
    parser.add_argument('--force_regenerate_index', action='store_true', help='Force regeneration of dataset index')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Optimizer weight decay')
    parser.add_argument('--beta', type=float, default=1.0, help='DPO beta hyperparameter')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Constraint slack epsilon')
    parser.add_argument('--eta_alpha', type=float, default=0.01, help='Learning rate for GRPO alpha weights (try smaller values like 0.01)') # Adjusted Default
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')

    # Training Mode
    parser.add_argument('--is_ref_training', action='store_true', help='Train the reference model (SFT) only.')
    parser.add_argument('--test_only', action='store_true', help='Run testing only using a checkpoint.')
    parser.add_argument('--test_checkpoint_path', type=str, default=None, help='Specific checkpoint for --test_only (defaults to best in checkpoint_dir)')

    # System
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU')
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency (batches)')

    # WandB
    parser.add_argument('--wandb_project', type=str, default="FairPO", help='WandB project name (disable if blank)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity')
    parser.add_argument('--wandb_id', type=str, default=None, help='WandB run ID to resume')
    parser.add_argument('--run_name', type=str, default="FairPO_training_full_32", help='Custom WandB run name (e.g., FairPO_b64_lr1e-5_eta0.01)')


    args = parser.parse_args()

    # Dynamic Run Name if not provided
    if not args.run_name:
        mode = "SFT" if args.is_ref_training else "FairPO"
        lr = args.learning_rate
        eta = args.eta_alpha if not args.is_ref_training else "N/A"
        args.run_name = f"{mode}_bs{args.batch_size}_lr{lr}_eta{eta}_{time.strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Generated run name: {args.run_name}")

    # Argument Validation
    if not args.is_ref_training and not args.test_only: # FairPO Training
        if not args.ref_cls_weights_path:
            logging.error("--- ERROR ---")
            logging.error("--ref_cls_weights_path is REQUIRED for FairPO training.")
            logging.error("Train SFT model first with --is_ref_training or provide the path.")
            sys.exit(1)
        elif not Path(args.ref_cls_weights_path).is_file():
             logging.error(f"Specified --ref_cls_weights_path '{args.ref_cls_weights_path}' not found.")
             sys.exit(1)
    elif args.test_only and not args.is_ref_training: # FairPO Testing
         if not args.ref_cls_weights_path or not Path(args.ref_cls_weights_path).is_file():
              logging.warning("--- WARNING ---")
              logging.warning("--ref_cls_weights_path not found or not specified for FairPO testing.")
              logging.warning("Test results may be inaccurate as the reference model might use initial weights.")
              logging.warning("-----------------")


    # Initialize Trainer
    trainer = ModelTrainer(args)

    # Run Training or Testing
    if args.test_only:
         trainer.test(args.test_checkpoint_path)
    else:
         trainer.train()
         # Run final test after training completion using the best checkpoint
         logging.info("Running final test using best checkpoint...")
         trainer.test() # Will use checkpoint_best.pth by default

if __name__ == '__main__':
    main()