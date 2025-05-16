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

if __name__ == "__main__": # Commented out for execution in interactive env
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    # Potentially make CuDNN deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
        logging.info(f"Using device: {self.device}")

        # --- Setup WandB ---
        self._setup_wandb()

        # --- Load Data ---
        logging.info("Loading datasets...")
        # Define privileged indices based on config (example: first 10 classes)
        # Adjust this based on actual COCO classes and fairness goals
        # coco.names has 80 classes (0-79)
        # Example: Let's make classes 0-19 privileged
        all_labels = list(range(80)) # Assuming 80 COCO classes
        # Convert comma-separated string from args to set of ints
        if config.privileged_indices:
             try:
                 privileged_indices_set = set(map(int, config.privileged_indices.split(',')))
                 # Validate indices
                 if not all(0 <= idx < 80 for idx in privileged_indices_set):
                     raise ValueError("Privileged indices must be between 0 and 79.")
             except ValueError as e:
                 logging.error(f"Invalid privileged indices format or value: {config.privileged_indices}. Error: {e}")
                 logging.error("Please provide a comma-separated list of integers between 0 and 79.")
                 sys.exit(1)
        else:
             privileged_indices_set = set(range(20)) # Default: first 20 classes privileged
             logging.warning(f"No privileged indices specified via --privileged_indices. Defaulting to: {sorted(list(privileged_indices_set))}")

        self.train_dataset = COCODatasetOnDemand(
            root_dir=config.coco_root,
            frac=config.train_frac,
            is_train=True,
            privileged_indices_set=privileged_indices_set,
            seed=seed,
            index_dir=config.index_dir,
            force_regenerate=config.force_regenerate_index
        )
        self.val_dataset = COCODatasetOnDemand(
            root_dir=config.coco_root,
            frac=config.val_frac,
            is_train=False, # Use validation split
            privileged_indices_set=privileged_indices_set,
            seed=seed,
            index_dir=config.index_dir,
            force_regenerate=config.force_regenerate_index
        )
        self.test_dataset = COCODatasetOnDemand( # Use val split also for testing here, or a dedicated test split if available
             root_dir=config.coco_root,
             frac=config.val_frac, # Reuse val fraction for test here
             is_train=False,
             privileged_indices_set=privileged_indices_set,
             seed=seed, # Use same seed for consistency if using same split
             index_dir=config.index_dir,
             force_regenerate=config.force_regenerate_index # Usually false for test
        )

        self.label_names = self.train_dataset.get_label_names()
        self.num_labels = len(self.label_names)
        self.privileged_indices = self.train_dataset.privileged_indices
        self.non_privileged_indices = self.train_dataset.non_privileged_indices

        logging.info(f"Number of training samples: {len(self.train_dataset)}")
        logging.info(f"Number of validation samples: {len(self.val_dataset)}")
        logging.info(f"Number of test samples: {len(self.test_dataset)}")
        logging.info(f"Number of labels: {self.num_labels}")
        logging.info(f"Privileged Indices: {self.privileged_indices}")
        logging.info(f"Non-Privileged Indices: {self.non_privileged_indices}")

        # --- Create DataLoaders ---
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            collate_fn=collate_fn_skip_none # Use the special collate function
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            collate_fn=collate_fn_skip_none
        )
        self.test_loader = DataLoader(
             self.test_dataset,
             batch_size=config.batch_size,
             shuffle=False,
             num_workers=config.num_workers,
             pin_memory=True if self.device == 'cuda' else False,
             collate_fn=collate_fn_skip_none
        )

        # --- Initialize Model ---
        logging.info(f"Initializing model: {config.model_name}")
        # Check if reference classifier weights exist
        ref_weights_path = Path(config.ref_cls_weights_path) if config.ref_cls_weights_path else None
        if not config.is_ref_training: # For FairPO training
            if ref_weights_path is None or not ref_weights_path.is_file():
                 logging.error(f"Reference classifier weights not found at '{config.ref_cls_weights_path}'.")
                 logging.error("FairPO training requires pre-trained reference classifier weights (trained via SFT).")
                 logging.error("You can train them first by running this script with --is_ref_training.")
                 sys.exit(1)
            else:
                 logging.info(f"Found reference classifier weights at: {ref_weights_path}")


        self.model = VisionModelForCLS(
            device=self.device,
            model_name=config.model_name,
            num_labels=self.num_labels,
            ref_cls_weights_path=str(ref_weights_path) if ref_weights_path else None,
            privileged_indices=self.privileged_indices,
            non_privileged_indices=self.non_privileged_indices,
            is_ref=config.is_ref_training, # Set based on whether we are training SFT or FairPO
            beta=config.beta,
            epsilon=config.epsilon,
            quant_config=None # TODO: Add BitsAndBytesConfig if needed and CUDA is available
        ).to(self.device)

        self.model.calc_num_params() # Print model summary

        # --- Optimizer ---
        # Only optimize the trainable classifier head parameters
        self.optimizer = optim.AdamW(
            self.model.model.classifier.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # --- GRPO Alpha Weights (only for FairPO training) ---
        if not config.is_ref_training:
            self.alpha_privileged = torch.tensor(0.5, device=self.device)
            self.alpha_non_privileged = torch.tensor(0.5, device=self.device)

        # --- Checkpoint Setup ---
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_metric = float('inf') # Lower is better for loss
        self.start_epoch = 0

        # --- Load Checkpoint if specified ---
        if config.resume_checkpoint:
            self._load_checkpoint(config.resume_checkpoint)

    def _setup_wandb(self):
        if self.config.wandb_project:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    config=vars(self.config), # Log all hyperparameters
                    name=self.config.run_name,
                    resume="allow", # Allow resuming runs
                    id=self.config.wandb_id # Pass a specific ID if resuming
                )
                wandb.watch(self.model, log='all', log_freq=self.config.log_freq) # Watch model gradients
                logging.info("WandB initialized.")
            except Exception as e:
                 logging.error(f"Failed to initialize WandB: {e}")
                 self.config.wandb_project = None # Disable wandb if init fails
        else:
            logging.info("WandB logging is disabled (no project name specified).")

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.model.classifier.state_dict(), # Save only classifier weights
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
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
            # Also save SFT weights with a specific name if training reference model
            if self.config.is_ref_training:
                 sft_filename = "sft_classifier_weights.pth"
                 sft_filepath = self.checkpoint_dir / sft_filename
                 # Save only the classifier state dict for SFT reference
                 torch.save(self.model.model.classifier.state_dict(), sft_filepath)
                 logging.info(f"Best SFT classifier weights saved to {sft_filepath}")
                 # Update config path for potential later FairPO runs
                 self.config.ref_cls_weights_path = str(sft_filepath)


    def _load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load classifier weights
            # Need to handle potential mismatches if model structure changed
            self.model.model.classifier.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_val_metric = checkpoint.get('best_val_metric', float('inf')) # Use get for backward compatibility

            if not self.config.is_ref_training and 'alpha_privileged' in checkpoint:
                 self.alpha_privileged = checkpoint['alpha_privileged'].to(self.device)
                 self.alpha_non_privileged = checkpoint['alpha_non_privileged'].to(self.device)
                 logging.info(f"Loaded alpha weights: Priv={self.alpha_privileged.item():.4f}, NonPriv={self.alpha_non_privileged.item():.4f}")

            logging.info(f"Resumed training from epoch {self.start_epoch}")
        else:
            logging.warning(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")


    def train(self):
        logging.info("Starting training...")
        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train() # Set model to training mode
            epoch_loss_priv = 0.0
            epoch_loss_non_priv = 0.0
            epoch_loss_total = 0.0 # For FairPO combined loss
            epoch_loss_sft = 0.0    # For SFT loss
            epoch_acc_priv = 0.0
            epoch_acc_non_priv = 0.0
            epoch_acc_overall = 0.0
            processed_batches = 0
            batch_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", unit="batch")
            start_time = time.time()

            for i, batch in enumerate(batch_iter):
                # Handle potentially empty batches from collate_fn_skip_none
                if not batch or 'pixels' not in batch or batch['pixels'].numel() == 0:
                     logging.warning(f"Skipping empty batch {i} in epoch {epoch+1}.")
                     continue

                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass through the model
                output_dict = self.model(pixels=pixels, labels=labels)

                if output_dict is None or 'loss' not in output_dict or output_dict['loss'] is None:
                     logging.warning(f"Skipping batch {i} due to model returning None or no loss.")
                     continue

                loss_components = output_dict['loss']
                acc_components = output_dict['acc']

                # --- Loss Calculation & Backpropagation ---
                if self.config.is_ref_training:
                    # --- SFT Training ---
                    loss = loss_components.get("loss", torch.tensor(0.0, device=self.device)) # Standard BCE loss
                    if torch.isnan(loss):
                        logging.warning(f"NaN loss detected during SFT training at epoch {epoch+1}, batch {i}. Skipping batch.")
                        continue
                    loss.backward()
                    epoch_loss_sft += loss.item()
                    log_payload = {"train/sft_loss": loss.item()}

                else:
                    # --- FairPO Training ---
                    loss_priv = loss_components.get("privileged", torch.tensor(0.0, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(0.0, device=self.device))

                    # Handle potential NaN losses before alpha update or backprop
                    if torch.isnan(loss_priv) or torch.isnan(loss_non_priv):
                        logging.warning(f"NaN loss detected (Priv: {loss_priv.item()}, NonPriv: {loss_non_priv.item()}) at epoch {epoch+1}, batch {i}. Skipping batch.")
                        continue

                    # GRPO Alpha Update (Mirror Ascent / Exponential Weighting)
                    with torch.no_grad(): # Don't track gradients for alpha update
                         # Use item() for exp to avoid tensor overhead if losses are scalar
                         new_alpha_priv = self.alpha_privileged * torch.exp(self.config.eta_alpha * loss_priv)
                         new_alpha_non_priv = self.alpha_non_privileged * torch.exp(self.config.eta_alpha * loss_non_priv)
                         
                         # Normalization constant Z
                         Z = new_alpha_priv + new_alpha_non_priv + 1e-8 # Add epsilon for stability

                         self.alpha_privileged = new_alpha_priv / Z
                         self.alpha_non_privileged = new_alpha_non_priv / Z

                    # Combined loss for backpropagation
                    total_loss = self.alpha_privileged * loss_priv + self.alpha_non_privileged * loss_non_priv

                    if torch.isnan(total_loss):
                        logging.warning(f"NaN total_loss detected after alpha weighting at epoch {epoch+1}, batch {i}. Skipping batch.")
                        continue

                    total_loss.backward()

                    # Accumulate epoch losses for logging
                    epoch_loss_priv += loss_priv.item()
                    epoch_loss_non_priv += loss_non_priv.item()
                    epoch_loss_total += total_loss.item()

                    log_payload = {
                         "train/privileged_loss": loss_priv.item(),
                         "train/non_privileged_loss": loss_non_priv.item(),
                         "train/combined_loss": total_loss.item(),
                         "train/alpha_privileged": self.alpha_privileged.item(),
                         "train/alpha_non_privileged": self.alpha_non_privileged.item(),
                    }

                # --- Optimizer Step ---
                # Optional: Gradient Clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.model.classifier.parameters(), self.config.grad_clip)
                self.optimizer.step()

                # --- Accuracy Accumulation ---
                epoch_acc_priv += acc_components.get("privileged", np.nan) # Use nan if not present
                epoch_acc_non_priv += acc_components.get("non_privileged", np.nan)
                epoch_acc_overall += acc_components.get("acc", np.nan)
                processed_batches += 1

                # --- Logging ---
                if i % self.config.log_freq == 0:
                    step = epoch * len(self.train_loader) + i
                    log_payload.update({
                        "train/accuracy_overall": acc_components.get("acc", np.nan),
                        "train/accuracy_privileged": acc_components.get("privileged", np.nan),
                        "train/accuracy_non_privileged": acc_components.get("non_privileged", np.nan),
                        "epoch": epoch + (i / len(self.train_loader)),
                        "step": step,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                    if self.config.wandb_project:
                        wandb.log(log_payload, step=step)

                    # Update tqdm progress bar
                    batch_iter.set_postfix(
                        loss=log_payload.get('train/combined_loss', log_payload.get('train/sft_loss', 0.0)),
                        acc=log_payload.get('train/accuracy_overall', 0.0),
                        alpha_P=log_payload.get('train/alpha_privileged', 0.0) # Show alpha_priv
                    )


            # --- End of Epoch ---
            epoch_duration = time.time() - start_time
            logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")

            if processed_batches == 0:
                 logging.warning(f"Epoch {epoch+1} finished without processing any batches. Check data pipeline.")
                 continue # Skip validation if no batches were processed

            # Calculate average epoch metrics
            avg_loss_priv = epoch_loss_priv / processed_batches if not self.config.is_ref_training else 0
            avg_loss_non_priv = epoch_loss_non_priv / processed_batches if not self.config.is_ref_training else 0
            avg_loss_total = epoch_loss_total / processed_batches if not self.config.is_ref_training else 0
            avg_loss_sft = epoch_loss_sft / processed_batches if self.config.is_ref_training else 0
            avg_acc_priv = epoch_acc_priv / processed_batches
            avg_acc_non_priv = epoch_acc_non_priv / processed_batches
            avg_acc_overall = epoch_acc_overall / processed_batches

            # --- Validation ---
            val_metrics = self._validate(epoch)
            avg_val_loss = val_metrics.get('val/loss_combined', val_metrics.get('val/sft_loss', float('inf')))

            # --- Log Epoch Metrics ---
            epoch_log_payload = {
                 "epoch": epoch + 1,
                 "train/epoch_duration_sec": epoch_duration,
                 "train/avg_accuracy_overall": avg_acc_overall,
                 "train/avg_accuracy_privileged": avg_acc_priv,
                 "train/avg_accuracy_non_privileged": avg_acc_non_priv,
            }
            if self.config.is_ref_training:
                 epoch_log_payload["train/avg_sft_loss"] = avg_loss_sft
            else:
                 epoch_log_payload.update({
                     "train/avg_privileged_loss": avg_loss_priv,
                     "train/avg_non_privileged_loss": avg_loss_non_priv,
                     "train/avg_combined_loss": avg_loss_total,
                     "train/final_alpha_privileged": self.alpha_privileged.item(),
                     "train/final_alpha_non_privileged": self.alpha_non_privileged.item(),
                 })
            # Add validation metrics to epoch log
            epoch_log_payload.update(val_metrics)

            if self.config.wandb_project:
                wandb.log(epoch_log_payload, step=(epoch + 1) * len(self.train_loader))

            # --- Checkpointing ---
            is_best = avg_val_loss < self.best_val_metric
            if is_best:
                self.best_val_metric = avg_val_loss
                logging.info(f"New best validation loss: {self.best_val_metric:.4f}")
            self._save_checkpoint(epoch, is_best=is_best)

        logging.info("Training finished.")
        if self.config.wandb_project:
            wandb.finish()


    def _validate(self, epoch):
        logging.info(f"Starting validation for epoch {epoch+1}...")
        self.model.eval() # Set model to evaluation mode
        total_val_loss_priv = 0.0
        total_val_loss_non_priv = 0.0
        total_val_loss_combined = 0.0 # FairPO
        total_val_loss_sft = 0.0     # SFT
        total_val_acc_priv = 0.0
        total_val_acc_non_priv = 0.0
        total_val_acc_overall = 0.0
        processed_batches = 0

        with torch.no_grad(): # Disable gradient calculations
            val_iter = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", unit="batch", leave=False)
            for batch in val_iter:
                if not batch or 'pixels' not in batch or batch['pixels'].numel() == 0:
                     continue

                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)

                output_dict = self.model(pixels=pixels, labels=labels)

                if output_dict is None or 'loss' not in output_dict or output_dict['loss'] is None:
                     continue

                loss_components = output_dict['loss']
                acc_components = output_dict['acc']

                # Accumulate metrics
                if self.config.is_ref_training:
                    loss = loss_components.get("loss", torch.tensor(0.0, device=self.device))
                    if not torch.isnan(loss):
                         total_val_loss_sft += loss.item()
                else:
                    loss_priv = loss_components.get("privileged", torch.tensor(0.0, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(0.0, device=self.device))
                    if not torch.isnan(loss_priv):
                         total_val_loss_priv += loss_priv.item()
                    if not torch.isnan(loss_non_priv):
                         total_val_loss_non_priv += loss_non_priv.item()
                    # Note: Combined loss uses training alphas, which might not be ideal for validation metric.
                    # We can calculate it just for consistency or focus on individual components.
                    # Let's use the final training alphas for reporting consistency
                    combined_loss = self.alpha_privileged * loss_priv + self.alpha_non_privileged * loss_non_priv
                    if not torch.isnan(combined_loss):
                         total_val_loss_combined += combined_loss.item()


                # Use np.nanmean perhaps later if needed, for now simple sum
                acc_priv = acc_components.get("privileged", np.nan)
                acc_non_priv = acc_components.get("non_privileged", np.nan)
                acc_overall = acc_components.get("acc", np.nan)

                if not np.isnan(acc_priv): total_val_acc_priv += acc_priv
                if not np.isnan(acc_non_priv): total_val_acc_non_priv += acc_non_priv
                if not np.isnan(acc_overall): total_val_acc_overall += acc_overall

                processed_batches += 1

        if processed_batches == 0:
             logging.warning("Validation finished without processing any batches.")
             return {} # Return empty dict if no validation happened

        # Calculate average validation metrics
        avg_val_loss_priv = total_val_loss_priv / processed_batches if not self.config.is_ref_training else 0
        avg_val_loss_non_priv = total_val_loss_non_priv / processed_batches if not self.config.is_ref_training else 0
        avg_val_loss_combined = total_val_loss_combined / processed_batches if not self.config.is_ref_training else 0
        avg_val_loss_sft = total_val_loss_sft / processed_batches if self.config.is_ref_training else 0
        avg_val_acc_priv = total_val_acc_priv / processed_batches
        avg_val_acc_non_priv = total_val_acc_non_priv / processed_batches
        avg_val_acc_overall = total_val_acc_overall / processed_batches

        logging.info(f"Validation Epoch {epoch+1} Results:")
        if self.config.is_ref_training:
             logging.info(f"  Avg Loss (SFT): {avg_val_loss_sft:.4f}")
        else:
             logging.info(f"  Avg Loss (Priv): {avg_val_loss_priv:.4f}")
             logging.info(f"  Avg Loss (Non-Priv): {avg_val_loss_non_priv:.4f}")
             logging.info(f"  Avg Loss (Combined): {avg_val_loss_combined:.4f}")
        logging.info(f"  Avg Acc (Overall): {avg_val_acc_overall:.4f}")
        logging.info(f"  Avg Acc (Priv): {avg_val_acc_priv:.4f}")
        logging.info(f"  Avg Acc (Non-Priv): {avg_val_acc_non_priv:.4f}")

        self.model.train() # Set back to training mode

        # Prepare metrics dictionary for logging
        val_metrics = {
            "val/accuracy_overall": avg_val_acc_overall,
            "val/accuracy_privileged": avg_val_acc_priv,
            "val/accuracy_non_privileged": avg_val_acc_non_priv,
        }
        if self.config.is_ref_training:
            val_metrics["val/sft_loss"] = avg_val_loss_sft
        else:
             val_metrics.update({
                 "val/loss_privileged": avg_val_loss_priv,
                 "val/loss_non_privileged": avg_val_loss_non_priv,
                 "val/loss_combined": avg_val_loss_combined, # Using training alphas
             })
        return val_metrics


    def test(self, checkpoint_path=None):
        logging.info("Starting testing...")
        # Load the best checkpoint if path is not provided
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_best.pth"
            if not checkpoint_path.is_file():
                 # Fallback to SFT weights if best FairPO checkpoint doesn't exist and we are testing FairPO
                 if not self.config.is_ref_training and self.config.ref_cls_weights_path:
                      sft_path = Path(self.config.ref_cls_weights_path)
                      if sft_path.is_file():
                           logging.warning(f"Best checkpoint not found at {checkpoint_path}. Loading SFT weights from {sft_path} for testing.")
                           # Load SFT weights into the main classifier for a baseline test
                           self.model.model.classifier.load_state_dict(torch.load(sft_path, map_location=self.device))
                           # Make sure ref classifier also has SFT weights if testing FairPO logic
                           if hasattr(self.model.model, 'ref_classifier'):
                                self.model.model.ref_classifier.load_state_dict(torch.load(sft_path, map_location=self.device))
                      else:
                          logging.error(f"Best checkpoint {checkpoint_path} not found, and SFT weights {sft_path} also not found. Cannot test.")
                          return {}
                 else:
                      logging.error(f"Best checkpoint not found at {checkpoint_path}. Cannot test.")
                      return {}

        if checkpoint_path.is_file():
             logging.info(f"Loading checkpoint for testing: {checkpoint_path}")
             # Custom loading logic for testing: only load model weights
             checkpoint = torch.load(checkpoint_path, map_location=self.device)
             if 'model_state_dict' in checkpoint:
                 self.model.model.classifier.load_state_dict(checkpoint['model_state_dict'])
                 logging.info("Loaded classifier weights from checkpoint.")
                 # Ensure reference classifier has weights if needed for FairPO test
                 if not self.config.is_ref_training and hasattr(self.model.model, 'ref_classifier'):
                      ref_weights_file = Path(self.config.ref_cls_weights_path) if self.config.ref_cls_weights_path else None
                      if ref_weights_file and ref_weights_file.is_file():
                           self.model.model.ref_classifier.load_state_dict(torch.load(ref_weights_file, map_location=self.device))
                           logging.info(f"Loaded reference weights from {ref_weights_file} for FairPO test.")
                      else:
                           logging.warning("Reference weights needed for FairPO test not found. Using initialized ref weights.")
             else:
                  # If it's just the state dict (like SFT weights)
                  self.model.model.classifier.load_state_dict(checkpoint)
                  logging.info("Loaded classifier weights directly from state dict file.")


        self.model.eval()
        total_test_loss_priv = 0.0
        total_test_loss_non_priv = 0.0
        total_test_loss_combined = 0.0
        total_test_loss_sft = 0.0
        total_test_acc_priv = 0.0
        total_test_acc_non_priv = 0.0
        total_test_acc_overall = 0.0
        processed_batches = 0

        with torch.no_grad():
            test_iter = tqdm(self.test_loader, desc="Testing", unit="batch", leave=False)
            for batch in test_iter:
                if not batch or 'pixels' not in batch or batch['pixels'].numel() == 0:
                     continue
                pixels = batch['pixels'].to(self.device)
                labels = batch['labels'].to(self.device)

                output_dict = self.model(pixels=pixels, labels=labels)

                if output_dict is None or 'loss' not in output_dict or output_dict['loss'] is None:
                     continue

                loss_components = output_dict['loss']
                acc_components = output_dict['acc']

                 # Accumulate metrics (similar to validation)
                if self.config.is_ref_training:
                    loss = loss_components.get("loss", torch.tensor(0.0, device=self.device))
                    if not torch.isnan(loss): total_test_loss_sft += loss.item()
                else:
                    loss_priv = loss_components.get("privileged", torch.tensor(0.0, device=self.device))
                    loss_non_priv = loss_components.get("non_privileged", torch.tensor(0.0, device=self.device))
                    if not torch.isnan(loss_priv): total_test_loss_priv += loss_priv.item()
                    if not torch.isnan(loss_non_priv): total_test_loss_non_priv += loss_non_priv.item()
                    # Use final training alphas if available, else default 0.5
                    alpha_p = self.alpha_privileged.item() if hasattr(self, 'alpha_privileged') else 0.5
                    alpha_np = self.alpha_non_privileged.item() if hasattr(self, 'alpha_non_privileged') else 0.5
                    combined_loss = alpha_p * loss_priv + alpha_np * loss_non_priv
                    if not torch.isnan(combined_loss): total_test_loss_combined += combined_loss.item()

                acc_priv = acc_components.get("privileged", np.nan)
                acc_non_priv = acc_components.get("non_privileged", np.nan)
                acc_overall = acc_components.get("acc", np.nan)

                if not np.isnan(acc_priv): total_test_acc_priv += acc_priv
                if not np.isnan(acc_non_priv): total_test_acc_non_priv += acc_non_priv
                if not np.isnan(acc_overall): total_test_acc_overall += acc_overall

                processed_batches += 1

        if processed_batches == 0:
            logging.error("Testing finished without processing any batches.")
            return {}

        # Calculate average test metrics
        avg_test_loss_priv = total_test_loss_priv / processed_batches if not self.config.is_ref_training else 0
        avg_test_loss_non_priv = total_test_loss_non_priv / processed_batches if not self.config.is_ref_training else 0
        avg_test_loss_combined = total_test_loss_combined / processed_batches if not self.config.is_ref_training else 0
        avg_test_loss_sft = total_test_loss_sft / processed_batches if self.config.is_ref_training else 0
        avg_test_acc_priv = total_test_acc_priv / processed_batches
        avg_test_acc_non_priv = total_test_acc_non_priv / processed_batches
        avg_test_acc_overall = total_test_acc_overall / processed_batches

        logging.info("--- Test Results ---")
        if self.config.is_ref_training:
             logging.info(f"  Avg Loss (SFT): {avg_test_loss_sft:.4f}")
        else:
             logging.info(f"  Avg Loss (Priv): {avg_test_loss_priv:.4f}")
             logging.info(f"  Avg Loss (Non-Priv): {avg_test_loss_non_priv:.4f}")
             logging.info(f"  Avg Loss (Combined): {avg_test_loss_combined:.4f}")
        logging.info(f"  Avg Acc (Overall): {avg_test_acc_overall:.4f}")
        logging.info(f"  Avg Acc (Priv): {avg_test_acc_priv:.4f}")
        logging.info(f"  Avg Acc (Non-Priv): {avg_test_acc_non_priv:.4f}")
        logging.info("--------------------")

        test_results = {
            "test/accuracy_overall": avg_test_acc_overall,
            "test/accuracy_privileged": avg_test_acc_priv,
            "test/accuracy_non_privileged": avg_test_acc_non_priv,
        }
        if self.config.is_ref_training:
            test_results["test/sft_loss"] = avg_test_loss_sft
        else:
             test_results.update({
                 "test/loss_privileged": avg_test_loss_priv,
                 "test/loss_non_privileged": avg_test_loss_non_priv,
                 "test/loss_combined": avg_test_loss_combined,
             })

        # Log test results to WandB if enabled
        if self.config.wandb_project and wandb.run is not None:
            # Use summary for final results if run is active
            for key, value in test_results.items():
                 wandb.summary[key] = value
            logging.info("Test results logged to WandB summary.")
        elif not self.config.wandb_project:
             logging.info("WandB disabled, not logging test results.")
        else: # wandb.run is None (already finished)
             logging.warning("WandB run already finished. Cannot log test results to WandB summary.")


        return test_results


def main():
    parser = argparse.ArgumentParser(description="Train FairPO Model for Multi-Label Classification")

    # --- Paths ---
    parser.add_argument('--coco_root', type=str, default="/home/soumen/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014", required=False, help='Root directory of the COCO dataset')
    parser.add_argument('--index_dir', type=str, default=None, help='Directory to store/load dataset index files (default: coco_root/.index_cache)')
    parser.add_argument('--checkpoint_dir', type=str, default='./output/ckpt/fairpo_model', help='Directory to save checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, default=False, help='Path to checkpoint to resume training from')
    parser.add_argument('--ref_cls_weights_path', type=str, default="/home/soumen/OML/FairPO/output/ckpt/ref_model/ckpt_ep_latest.pth", help='Path to SFT pre-trained reference classifier weights (REQUIRED for FairPO)')

    # --- Model & Data ---
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224', help='Name of the Vision Transformer model from Hugging Face')
    parser.add_argument('--train_frac', type=float, default=1.0, help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--val_frac', type=float, default=1.0, help='Fraction of validation data to use (0.0 to 1.0)')
    parser.add_argument('--privileged_indices', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19", help='Comma-separated list of privileged label indices (e.g., "0,5,10")')
    parser.add_argument('--force_regenerate_index', action='store_true', help='Force regeneration of dataset index files')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Optimizer learning rate (eta_params)')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Optimizer weight decay')
    parser.add_argument('--beta', type=float, default=1.0, help='DPO beta hyperparameter for privileged loss')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Constraint slack epsilon for non-privileged loss')
    parser.add_argument('--eta_alpha', type=float, default=0.1, help='Learning rate for GRPO alpha weights update')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')

    # --- Training Mode ---
    parser.add_argument('--is_ref_training', action='store_true', help='Set this flag to train the reference model (SFT) instead of FairPO.')
    parser.add_argument('--test_only', action='store_true', help='Set this flag to only run testing using a checkpoint.')
    parser.add_argument('--test_checkpoint_path', type=str, default=None, help='Path to specific checkpoint for --test_only mode (defaults to best checkpoint)')


    # --- System ---
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU even if CUDA is available')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency (in batches) for logging training progress')

    # --- WandB ---
    parser.add_argument('--wandb_project', type=str, default="FairPO", help='WandB project name (leave blank to disable)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team)')
    parser.add_argument('--wandb_id', type=str, default=None, help='WandB run ID to resume a specific run')
    parser.add_argument('--run_name', type=str, default="FairPO_training_full_64", help='Custom name for the WandB run')

    args = parser.parse_args()

    # --- Argument Validation ---
    if not args.is_ref_training and args.ref_cls_weights_path is None and not args.resume_checkpoint and not args.test_only:
         logging.warning("--- IMPORTANT ---")
         logging.warning("Running FairPO training (--is_ref_training not set) but --ref_cls_weights_path is not specified.")
         logging.warning("FairPO requires pre-trained SFT weights for the reference classifier.")
         logging.warning("Proceeding without reference weights - the reference model will use INITIALIZED weights, which is likely NOT desired for FairPO.")
         logging.warning("Consider training SFT first with --is_ref_training or provide the path via --ref_cls_weights_path.")
         logging.warning("-----------------")
         # Allow proceeding but with the warning. The model init handles None path internally.
    elif not args.is_ref_training and args.ref_cls_weights_path:
         if not Path(args.ref_cls_weights_path).is_file():
             logging.error(f"Specified reference classifier weights path '{args.ref_cls_weights_path}' does not exist or is not a file.")
             sys.exit(1)

    # --- Initialize Trainer ---
    trainer = ModelTrainer(args)

    # --- Run Training or Testing ---
    if args.test_only:
         trainer.test(args.test_checkpoint_path)
    else:
         trainer.train()
         # Run final test after training completion
         trainer.test() # Uses best checkpoint by default

if __name__ == '__main__':
    main()