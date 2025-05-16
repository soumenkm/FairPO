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
from typing import Dict, Any, Tuple, Optional

from models import VisionModelForCLS
from dataset import COCODatasetOnDemand, NUSWIDEDatasetOnDemand, collate_fn_skip_none

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class ModelTrainer:
    """
    Trainer class to manage model training, validation, checkpointing, and logging.
    Supports both reference model (SFT) training and FairPO training with privileged/non-privileged losses.
    """

    def __init__(self, config):
        """
        Initialize ModelTrainer with configuration, datasets, model, optimizer, and logging.

        Args:
            config: Namespace object containing configuration parameters.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
        logging.info(f"Using device: {self.device}")
        self.train_step = 0
        self.val_step = 0
        self.start_epoch = 0

        run_name = self._get_run_name()
        self.config.run_name = run_name
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.wandb_project / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints will be saved in: {self.checkpoint_dir}")

        logging.info(f"Loading dataset: {config.dataset_name.upper()}")
        privileged_indices_set = set(map(int, config.privileged_indices.split(','))) if config.privileged_indices else set()

        if config.dataset_name == "coco":
            dataset_root = Path(config.coco_root)
            index_dir_path = Path(config.index_dir) if config.index_dir else dataset_root / '.coco_index_cache'
        elif config.dataset_name == "nuswide":
            dataset_root = Path(config.nus_root)
            index_dir_path = Path(config.index_dir) if config.index_dir else dataset_root / '.nuswide_index_cache'
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_name}")

        index_dir_path.mkdir(parents=True, exist_ok=True)
        self.config.index_dir = index_dir_path

        if config.dataset_name == "coco":
            self.train_dataset = COCODatasetOnDemand(
                root_dir=str(dataset_root), frac=config.train_frac, split_name="train",
                privileged_indices_set=privileged_indices_set, seed=seed,
                index_dir=str(self.config.index_dir), force_regenerate=config.force_regenerate_index
            )
            self.val_dataset = COCODatasetOnDemand(
                root_dir=str(dataset_root), frac=config.val_frac, split_name="val",
                privileged_indices_set=privileged_indices_set, seed=seed,
                index_dir=str(self.config.index_dir), force_regenerate=config.force_regenerate_index
            )
        else:
            self.train_dataset = NUSWIDEDatasetOnDemand(
                root_dir=str(dataset_root), frac=config.train_frac, split_name="train",
                privileged_indices_set=privileged_indices_set, seed=seed,
                index_dir=str(self.config.index_dir), force_regenerate=config.force_regenerate_index
            )
            self.val_dataset = NUSWIDEDatasetOnDemand(
                root_dir=str(dataset_root), frac=config.val_frac, split_name="test",
                privileged_indices_set=privileged_indices_set, seed=seed,
                index_dir=str(self.config.index_dir), force_regenerate=config.force_regenerate_index
            )

        logging.info(f"Using index directory: {self.config.index_dir}")

        self.label_names = self.train_dataset.get_label_names()
        self.num_labels = len(self.label_names)
        self.privileged_indices = sorted(list(self.train_dataset.privileged_indices))
        self.non_privileged_indices = sorted(list(self.train_dataset.non_privileged_indices))

        logging.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        logging.info(f"Num labels: {self.num_labels}")
        logging.info(f"Privileged Indices ({len(self.privileged_indices)}): {self.privileged_indices}")
        logging.info(f"Non-Privileged Indices ({len(self.non_privileged_indices)}): {self.non_privileged_indices}")

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
        if not config.is_ref_training:
            if ref_weights_path is None or not ref_weights_path.is_file():
                logging.error(f"Reference classifier weights REQUIRED for FairPO training but not found: '{config.ref_cls_weights_path}'.")
                logging.error("Please provide a valid path to SFT model weights using --ref_cls_weights_path.")
                sys.exit(1)
            logging.info(f"Using reference classifier weights for FairPO: {ref_weights_path}")
        elif ref_weights_path and ref_weights_path.is_file():
            logging.warning(f"Reference weights path '{ref_weights_path}' provided, but ignored because --is_ref_training is True.")
            ref_weights_path = None

        self.model = VisionModelForCLS(
            device=self.device, model_name=config.model_name, num_labels=self.num_labels,
            ref_cls_weights_path=str(ref_weights_path) if ref_weights_path else None,
            privileged_indices=self.privileged_indices,
            non_privileged_indices=self.non_privileged_indices,
            is_ref=config.is_ref_training,
            loss_type=config.loss_type,
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
            self.loss_priv_ema = torch.tensor(0.0, device=self.device)
            self.loss_non_priv_ema = torch.tensor(0.0, device=self.device)
            self.num_updates_priv_loss_ema = 0
            self.num_updates_non_priv_loss_ema = 0
            self.alpha_privileged = torch.tensor(0.5, device=self.device)
            self.alpha_non_privileged = torch.tensor(0.5, device=self.device)

        self._setup_wandb()

    def _calculate_norm(self, model_params, norm_type=2.0):
        """
        Calculate the norm of the model parameters.

        Args:
            model_params: Iterable of model parameters.
            norm_type: Type of norm to calculate (default 2).

        Returns:
            Total norm value as a tensor.
        """
        total_norm = torch.tensor(0.0, device=self.device)
        for p in model_params:
            if p is not None:
                param_norm = p.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)

    def _calculate_grad_norm(self, model_params, norm_type=2.0):
        """
        Calculate the gradient norm of model parameters.

        Args:
            model_params: Iterable of model parameters.
            norm_type: Type of norm to calculate (default 2).

        Returns:
            Total gradient norm as a tensor.
        """
        total_norm = torch.tensor(0.0, device=self.device)
        params_to_norm = [p for p in model_params if p is not None and p.grad is not None]
        if not params_to_norm:
            return total_norm
        for p in params_to_norm:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)

    def _setup_wandb(self):
        """
        Initialize WandB logging if enabled in config.
        """
        if self.config.is_wandb:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.run_name,
                    config=vars(self.config)
                )
                wandb.watch(self.model, log="all", log_freq=100)
                wandb.define_metric("train/step"); wandb.define_metric("val/step")
                wandb.define_metric("train/*", step_metric="train/step")
                wandb.define_metric("val/*", step_metric="val/step")
                wandb.define_metric("epoch", step_metric="train/step")
                wandb.define_metric("epoch", step_metric="val/step")
                logging.info(f"WandB initialized for project '{self.config.wandb_project}', run '{self.config.run_name}'.")
            except Exception as e:
                logging.error(f"Failed to initialize WandB: {e}")
                self.config.is_wandb = False
        else:
            logging.info("WandB disabled.")

    def _save_checkpoint(self, epoch, is_best=False):
        """
        Save checkpoint of the model, optimizer states, and training progress.

        Args:
            epoch: Current epoch number.
            is_best: Whether this checkpoint is the best so far.
        """
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'val_step': self.val_step,
            "config": vars(self.config)
        }
        if not self.config.is_ref_training:
            checkpoint_data['alpha_privileged'] = self.alpha_privileged
            checkpoint_data['alpha_non_privileged'] = self.alpha_non_privileged
            checkpoint_data['loss_priv_ema'] = self.loss_priv_ema
            checkpoint_data['loss_non_priv_ema'] = self.loss_non_priv_ema
            checkpoint_data['num_updates_priv_loss_ema'] = self.num_updates_priv_loss_ema
            checkpoint_data['num_updates_non_priv_loss_ema'] = self.num_updates_non_priv_loss_ema

        filename = f"checkpoint_epoch_{epoch+1}.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint_data, filepath)
        logging.info(f"Checkpoint saved: {filepath}")

        if is_best:
            best_filepath = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint_data, best_filepath)
            logging.info(f"Best checkpoint saved: {best_filepath}")

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Process a single batch of data through the model.

        Args:
            batch: Batch dictionary containing 'pixels' and 'labels'.

        Returns:
            Tuple of model output dictionary and batch size, or None if invalid batch.
        """
        if batch is None or 'pixels' not in batch or batch['pixels'].numel() == 0:
            logging.warning("Skipping empty or invalid batch.")
            return None

        pixels = batch['pixels'].to(self.device)
        labels = batch['labels'].to(self.device)
        current_batch_size = pixels.size(0)
        output_dict = self.model(pixels=pixels, labels=labels)
        return output_dict, current_batch_size

    def _update_epoch_metrics(self, epoch_metrics: Dict[str, Any], results_dict: Dict[str, Any], batch_size: int):
        """
        Accumulate batch results into epoch-level metrics.

        Args:
            epoch_metrics: Dictionary accumulating epoch sums.
            results_dict: Dictionary of results from current batch.
            batch_size: Number of samples in the batch.
        """
        loss_comp = results_dict.get('loss')
        if loss_comp:
            if self.config.is_ref_training:
                epoch_metrics['loss_sft'] += loss_comp.get('loss', torch.tensor(0.0)).item() * batch_size
            else:
                priv_loss_val = loss_comp.get('privileged', torch.tensor(0.0)).item()
                non_priv_loss_val = loss_comp.get('non_privileged', torch.tensor(0.0)).item()
                combined_loss_val = loss_comp.get('combined_loss_for_opt', torch.tensor(0.0)).item()

                epoch_metrics['loss_priv'] += priv_loss_val * batch_size
                epoch_metrics['loss_non_priv'] += non_priv_loss_val * batch_size
                epoch_metrics['loss_total'] += combined_loss_val * batch_size

        acc_comp = results_dict.get('acc')
        if acc_comp:
            epoch_metrics['acc_overall'] += acc_comp.get('acc', np.nan) * batch_size
            epoch_metrics['acc_priv'] += acc_comp.get('privileged', np.nan) * batch_size
            epoch_metrics['acc_non_priv'] += acc_comp.get('non_privileged', np.nan) * batch_size
        
        em_comp = results_dict.get('em')
        if em_comp:
            epoch_metrics['em'] += em_comp.get('em', np.nan) * batch_size
            epoch_metrics['em_priv'] += em_comp.get('em_privileged', np.nan) * batch_size
            epoch_metrics['em_non_priv'] += em_comp.get('em_non_privileged', np.nan) * batch_size

        f1_comp = results_dict.get('f1')
        if f1_comp:
            epoch_metrics['f1'] += f1_comp.get('f1', np.nan) * batch_size
            epoch_metrics['f1_priv'] += f1_comp.get('f1_privileged', np.nan) * batch_size
            epoch_metrics['f1_non_priv'] += f1_comp.get('f1_non_privileged', np.nan) * batch_size

        map_comp = results_dict.get('map')
        if map_comp:
            epoch_metrics['map_overall'] += map_comp.get('mAP', np.nan) * batch_size
            epoch_metrics['map_priv'] += map_comp.get('mAP_privileged', np.nan) * batch_size
            epoch_metrics['map_non_priv'] += map_comp.get('mAP_non_privileged', np.nan) * batch_size

    def _perform_optimization_step(self, loss: torch.Tensor) -> Optional[float]:
        """
        Perform backward pass and optimizer step with optional gradient clipping.

        Args:
            loss: Loss tensor to backpropagate.

        Returns:
            Gradient norm as a float.
        """
        loss.backward()
        grad_norm = self._calculate_grad_norm(self.trainable_params)

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.grad_clip)

        self.optimizer.step()
        return grad_norm.item()

    def _log_wandb_train_batch_metrics(self, results_dict: Dict[str, Any], grad_norm: Optional[float], param_norm: float, epoch: int, batch_idx: int, num_batches: int):
        """
        Log batch training metrics to WandB.

        Args:
            results_dict: Dictionary of batch results.
            grad_norm: Gradient norm.
            param_norm: Parameter norm.
            epoch: Current epoch number.
            batch_idx: Index of current batch.
            num_batches: Total batches in epoch.
        """
        if not self.config.is_wandb:
            return

        batch_log = {'train/step': self.train_step, 'epoch': epoch + (batch_idx + 1) / num_batches}

        loss_comp = results_dict.get('loss')
        if loss_comp:
            if self.config.is_ref_training:
                batch_log['train/sft_loss'] = loss_comp.get('loss', torch.tensor(np.nan)).item()
            else:
                batch_log['train/privileged_loss_batch_avg'] = loss_comp.get('privileged', torch.tensor(np.nan)).item()
                batch_log['train/non_privileged_loss_batch_avg'] = loss_comp.get('non_privileged', torch.tensor(np.nan)).item()
                batch_log['train/combined_loss_for_opt'] = loss_comp.get('combined_loss_for_opt', torch.tensor(np.nan)).item()
                batch_log["train/alpha_p"] = self.alpha_privileged.item()
                batch_log["train/alpha_np"] = self.alpha_non_privileged.item()
                batch_log["train/loss_priv_ema"] = self.loss_priv_ema.item()
                batch_log["train/loss_non_priv_ema"] = self.loss_non_priv_ema.item()

        acc_comp = results_dict.get('acc')
        if acc_comp:
            batch_log['train/accuracy_overall'] = acc_comp.get('acc', np.nan)
            batch_log['train/accuracy_privileged'] = acc_comp.get('privileged', np.nan)
            batch_log['train/accuracy_non_privileged'] = acc_comp.get('non_privileged', np.nan)

        em_comp = results_dict.get('em')
        if em_comp:
            batch_log['train/exact_match_overall'] = em_comp.get('em', np.nan)
            batch_log['train/exact_match_privileged'] = em_comp.get('em_privileged', np.nan)
            batch_log['train/exact_match_non_privileged'] = em_comp.get('em_non_privileged', np.nan)

        f1_comp = results_dict.get('f1')
        if f1_comp:
            batch_log['train/f1_overall'] = f1_comp.get('f1', np.nan)
            batch_log['train/f1_privileged'] = f1_comp.get('f1_privileged', np.nan)
            batch_log['train/f1_non_privileged'] = f1_comp.get('f1_non_privileged', np.nan)

        map_comp = results_dict.get('map')
        if map_comp:
            batch_log['train/map_overall'] = map_comp.get('mAP', np.nan)
            batch_log['train/map_privileged'] = map_comp.get('mAP_privileged', np.nan)
            batch_log['train/map_non_privileged'] = map_comp.get('mAP_non_privileged', np.nan)

        if grad_norm is not None:
            batch_log['train/grad_norm'] = grad_norm
        batch_log['train/param_norm'] = param_norm
        batch_log['train/learning_rate'] = self.optimizer.param_groups[0]['lr']

        wandb.log({k: v for k, v in batch_log.items() if not (isinstance(v, float) and np.isnan(v))})

    def _calculate_epoch_averages(self, epoch_metrics: Dict[str, Any], processed_items: int) -> Dict[str, float]:
        """
        Calculate averages of accumulated epoch metrics.

        Args:
            epoch_metrics: Dictionary of summed metrics.
            processed_items: Number of samples processed in epoch.

        Returns:
            Dictionary of averaged metrics.
        """
        avg_metrics = {}
        if processed_items == 0:
            logging.warning("No items processed in epoch, metrics will be NaN.")
            for k_template in ['loss_sft', 'loss_priv', 'loss_non_priv', 'loss_total',
                               'acc_overall', 'acc_priv', 'acc_non_priv',
                               'em', 'em_priv', 'em_non_priv',
                               'f1', 'f1_priv', 'f1_non_priv',
                               'map_overall', 'map_priv', 'map_non_priv']:
                avg_metrics[f'avg_{k_template}'] = np.nan
            return avg_metrics

        for key, value in epoch_metrics.items():
            avg_metrics[f'avg_{key}'] = value / processed_items if not np.isnan(value) else np.nan

        return avg_metrics

    def train(self):
        """
        Main training loop that manages training, validation, checkpointing, early stopping, and logging.
        """
        logging.info("Starting training...")
        best_val_metric = float('inf')
        no_improvement_epochs = 0

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            epoch_metrics = {
                'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
                'acc_overall': 0.0, 'acc_priv': 0.0, 'acc_non_priv': 0.0,
                'em': 0.0, 'em_priv': 0.0, 'em_non_priv': 0.0,
                'f1': 0.0, 'f1_priv': 0.0, 'f1_non_priv': 0.0,
                'map_overall': 0.0, 'map_priv': 0.0, 'map_non_priv': 0.0
            }
            processed_items_train = 0
            num_batches = len(self.train_loader)

            batch_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", unit="batch")
            start_time = time.time()

            for i, batch in enumerate(batch_iter):
                if batch is None or 'pixels' not in batch or batch['pixels'].numel() == 0:
                    logging.warning("Skipping empty or invalid batch in training loop.")
                    continue
                current_batch_size = batch['pixels'].size(0)

                processed_batch_info = self._process_batch(batch)
                if processed_batch_info is None:
                    continue

                results_dict, _ = processed_batch_info
                loss_components = results_dict['loss']
                self.optimizer.zero_grad()

                if self.config.is_ref_training:
                    combined_loss_for_opt = loss_components.get("loss", torch.tensor(0.0, device=self.device))
                else:
                    loss_priv_batch_avg = loss_components.get("privileged", torch.tensor(0.0, device=self.device))
                    loss_non_priv_batch_avg = loss_components.get("non_privileged", torch.tensor(0.0, device=self.device))

                    current_loss_p_detached = loss_priv_batch_avg.detach()
                    current_loss_np_detached = loss_non_priv_batch_avg.detach()

                    if self.num_updates_priv_loss_ema == 0:
                        scaled_exp_term_p = current_loss_p_detached
                        self.loss_priv_ema = current_loss_p_detached
                    else:
                        scaled_exp_term_p = (current_loss_p_detached - self.loss_priv_ema) / \
                                            (torch.abs(self.loss_priv_ema) + self.config.delta_scaling)
                        self.loss_priv_ema = self.config.ema_decay * self.loss_priv_ema + \
                                             (1 - self.config.ema_decay) * current_loss_p_detached
                    self.num_updates_priv_loss_ema += 1

                    if self.num_updates_non_priv_loss_ema == 0:
                        scaled_exp_term_np = current_loss_np_detached
                        self.loss_non_priv_ema = current_loss_np_detached
                    else:
                        scaled_exp_term_np = (current_loss_np_detached - self.loss_non_priv_ema) / \
                                             (torch.abs(self.loss_non_priv_ema) + self.config.delta_scaling)
                        self.loss_non_priv_ema = self.config.ema_decay * self.loss_non_priv_ema + \
                                               (1 - self.config.ema_decay) * current_loss_np_detached
                    self.num_updates_non_priv_loss_ema += 1

                    with torch.no_grad():
                        exp_arg_p = (self.config.eta_alpha * scaled_exp_term_p).clamp(-10, 10)
                        exp_arg_np = (self.config.eta_alpha * scaled_exp_term_np).clamp(-10, 10)

                        new_alpha_priv = self.alpha_privileged * torch.exp(exp_arg_p)
                        new_alpha_non_priv = self.alpha_non_privileged * torch.exp(exp_arg_np)
                        Z = new_alpha_priv + new_alpha_non_priv + 1e-8
                        self.alpha_privileged = new_alpha_priv / Z
                        self.alpha_non_privileged = new_alpha_non_priv / Z

                    combined_loss_for_opt = self.alpha_privileged * loss_priv_batch_avg + \
                                            self.alpha_non_privileged * loss_non_priv_batch_avg
                    results_dict['loss']['combined_loss_for_opt'] = combined_loss_for_opt

                grad_norm_val = self._perform_optimization_step(combined_loss_for_opt)

                self.train_step += 1
                param_norm_val = self._calculate_norm(self.trainable_params).item()

                self._update_epoch_metrics(epoch_metrics, results_dict, current_batch_size)
                processed_items_train += current_batch_size

                if self.config.is_wandb:
                    self._log_wandb_train_batch_metrics(results_dict, grad_norm_val, param_norm_val, epoch, i, num_batches)

                tqdm_postfix = {
                    'loss': combined_loss_for_opt.item(),
                    'acc': results_dict.get('acc', {}).get('acc', np.nan),
                    'step': self.train_step
                }
                if not self.config.is_ref_training:
                    tqdm_postfix['alpha_p'] = self.alpha_privileged.item()
                batch_iter.set_postfix(**{k: f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else (str(v) if isinstance(v, float) and np.isnan(v) else v) for k,v in tqdm_postfix.items()})

            epoch_duration = time.time() - start_time
            logging.info(f"Epoch {epoch+1} training completed in {epoch_duration:.2f}s")

            avg_train_metrics = self._calculate_epoch_averages(epoch_metrics, processed_items_train)
            avg_train_metrics_wandb = {f"train/{k.replace('avg_', '')}": v for k, v in avg_train_metrics.items()}

            if self.config.is_wandb:
                wandb.log({**avg_train_metrics_wandb, 'epoch': epoch + 1, 'train/step': self.train_step})

            avg_val_metrics_dict_prefixed = self._validate(epoch)

            if self.config.is_ref_training:
                current_val_metric_for_best = avg_val_metrics_dict_prefixed.get('val/loss_sft', float('inf'))
                is_best = current_val_metric_for_best < best_val_metric
            else:
                current_val_metric_for_best = -avg_val_metrics_dict_prefixed.get('val/map_priv', float('-inf'))
                is_best = current_val_metric_for_best < best_val_metric

            if is_best:
                best_val_metric = current_val_metric_for_best
                logging.info(f"New best validation metric achieved: {best_val_metric:.4f}")
                self._save_checkpoint(epoch, is_best=True)
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                logging.info(f"Validation metric did not improve. Current: {current_val_metric_for_best:.4f}, Best: {best_val_metric:.4f}. No improvement streak: {no_improvement_epochs}")

            self._save_checkpoint(epoch, is_best=False)

            self._log_epoch_summary_console(epoch, avg_train_metrics, mode='train')
            self._log_epoch_summary_console(epoch, avg_val_metrics_dict_prefixed, mode='val')

            if self.config.early_stopping_patience > 0 and no_improvement_epochs >= self.config.early_stopping_patience:
                logging.info(f"Early stopping triggered after {no_improvement_epochs} epochs without improvement.")
                break

        logging.info("Training finished.")
        if self.config.is_wandb:
            logging.info("Finishing WandB run.")
            wandb.summary['best_val_metric_achieved'] = best_val_metric
            wandb.finish()

    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Perform validation over the entire validation set.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of averaged validation metrics with 'val/' prefix.
        """
        logging.info(f"Starting validation for epoch {epoch+1}...")
        self.model.eval()
        val_epoch_metrics = {
            'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, 'loss_sft': 0.0,
            'acc_overall': 0.0, 'acc_priv': 0.0, 'acc_non_priv': 0.0,
            'em': 0.0, 'em_priv': 0.0, 'em_non_priv': 0.0,
            'f1': 0.0, 'f1_priv': 0.0, 'f1_non_priv': 0.0,
            'map_overall': 0.0, 'map_priv': 0.0, 'map_non_priv': 0.0
        }
        processed_items_val = 0

        with torch.no_grad():
            val_iter = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", unit="batch", leave=False)
            for batch in val_iter:
                if batch is None or 'pixels' not in batch or batch['pixels'].numel() == 0:
                    logging.warning("Skipping empty or invalid batch in validation loop.")
                    continue
                current_batch_size = batch['pixels'].size(0)

                processed_batch_info = self._process_batch(batch)
                if processed_batch_info is None:
                    continue
                results_dict, _ = processed_batch_info

                if not self.config.is_ref_training and 'loss' in results_dict and results_dict['loss'] is not None:
                    loss_priv_val = results_dict['loss'].get('privileged', torch.tensor(0.0, device=self.device))
                    loss_non_priv_val = results_dict['loss'].get('non_privileged', torch.tensor(0.0, device=self.device))
                    results_dict['loss']['combined_loss_for_opt'] = self.alpha_privileged * loss_priv_val + self.alpha_non_privileged * loss_non_priv_val

                self._update_epoch_metrics(val_epoch_metrics, results_dict, current_batch_size)
                processed_items_val += current_batch_size

        avg_val_metrics_raw = self._calculate_epoch_averages(val_epoch_metrics, processed_items_val)
        avg_val_metrics_prefixed = {f"val/{k.replace('avg_', '')}": v for k, v in avg_val_metrics_raw.items()}

        self.val_step += 1

        if self.config.is_wandb:
            self._log_wandb_validation_summary(epoch, avg_val_metrics_prefixed)

        self.model.train()
        return avg_val_metrics_prefixed

    def _log_epoch_summary_console(self, epoch: int, avg_epoch_metrics_dict: Dict[str, float], mode: str):
        """
        Log average epoch metrics to console.

        Args:
            epoch: Epoch number.
            avg_epoch_metrics_dict: Dictionary of averaged metrics.
            mode: Either 'train' or 'val'.
        """
        is_val_mode = mode == 'val'
        key_prefix = "val/" if is_val_mode else "avg_"
        title_prefix = "Training" if mode == 'train' else "Validation"
        step_info = f"(Train Step {self.train_step})" if mode == 'train' else f"(Val Step {self.val_step})"
        logging.info(f"--- {title_prefix} Epoch {epoch+1} Results {step_info} ---")

        if self.config.is_ref_training:
            logging.info(f"  Avg Loss (SFT): {avg_epoch_metrics_dict.get(f'{key_prefix}loss_sft', np.nan):.4f}")
        else:
            logging.info(f"  Avg Loss (Priv Actual Avg): {avg_epoch_metrics_dict.get(f'{key_prefix}loss_priv', np.nan):.4f}")
            logging.info(f"  Avg Loss (Non-Priv Actual Avg): {avg_epoch_metrics_dict.get(f'{key_prefix}loss_non_priv', np.nan):.4f}")
            logging.info(f"  Avg Loss (Combined Opt/Eval): {avg_epoch_metrics_dict.get(f'{key_prefix}loss_total', np.nan):.4f}")
            if mode == 'train':
                logging.info(f"  Alphas (P/NP): {self.alpha_privileged.item():.3f} / {self.alpha_non_privileged.item():.3f}")
                logging.info(f"  Loss EMAs (P/NP): {self.loss_priv_ema.item():.4f} / {self.loss_non_priv_ema.item():.4f}")

        logging.info(f"  Avg Acc (Overall): {avg_epoch_metrics_dict.get(f'{key_prefix}acc_overall', np.nan):.4f}")
        logging.info(f"  Avg Acc (Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}acc_priv', np.nan):.4f}")
        logging.info(f"  Avg Acc (Non-Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}acc_non_priv', np.nan):.4f}")

        logging.info(f"  Avg EM (Overall): {avg_epoch_metrics_dict.get(f'{key_prefix}em', np.nan):.4f}")
        logging.info(f"  Avg EM (Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}em_priv', np.nan):.4f}")
        logging.info(f"  Avg EM (Non-Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}em_non_priv', np.nan):.4f}")

        logging.info(f"  Avg F1 (Overall): {avg_epoch_metrics_dict.get(f'{key_prefix}f1', np.nan):.4f}")
        logging.info(f"  Avg F1 (Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}f1_priv', np.nan):.4f}")
        logging.info(f"  Avg F1 (Non-Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}f1_non_priv', np.nan):.4f}")

        logging.info(f"  Avg mAP (Overall): {avg_epoch_metrics_dict.get(f'{key_prefix}map_overall', np.nan):.4f}")
        logging.info(f"  Avg mAP (Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}map_priv', np.nan):.4f}")
        logging.info(f"  Avg mAP (Non-Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}map_non_priv', np.nan):.4f}")
        logging.info(f"---------------------------------------------")

    def _log_wandb_validation_summary(self, epoch: int, val_metrics_prefixed: Dict[str, float]):
        """
        Log validation summary metrics to WandB.

        Args:
            epoch: Current epoch number.
            val_metrics_prefixed: Dictionary of validation metrics with 'val/' prefix.
        """
        if not self.config.is_wandb:
            return
        log_payload = {**val_metrics_prefixed}
        log_payload['train/step'] = self.train_step
        log_payload['val/step'] = self.val_step
        log_payload['epoch'] = epoch + 1
        wandb.log({k: v for k, v in log_payload.items() if not (isinstance(v, float) and np.isnan(v))})

    def _get_run_name(self) -> str:
        """
        Generate or retrieve the run name for logging and checkpointing.

        Returns:
            A sanitized string run name.
        """
        if self.config.run_name and self.config.run_name != 'generate':
            return self.config.run_name

        dataset_prefix = self.config.dataset_name.upper()
        mode = "SFT" if self.config.is_ref_training else "FairPO"
        loss_part = f"_{self.config.loss_type}" if not self.config.is_ref_training and self.config.loss_type else ""
        lr_str = f"lr{self.config.learning_rate:.0e}"
        ep_str = f"ep{self.config.epochs}"
        frac_str = f"frac{self.config.train_frac:.2f}"

        fairpo_params = ""
        if not self.config.is_ref_training:
            eta_str = f"eta{self.config.eta_alpha}"
            eps_str = f"_eps{self.config.epsilon}"
            beta_str = f"_beta{self.config.beta}"
            ema_decay_str = f"_ema{self.config.ema_decay}"
            delta_scale_str = f"_deltaS{self.config.delta_scaling}"
            fairpo_params = f"{eta_str}{eps_str}{beta_str}{ema_decay_str}{delta_scale_str}_"

        run_name = f"{dataset_prefix}_{mode}{loss_part}_{lr_str}_{fairpo_params}{frac_str}_{ep_str}"
        run_name = run_name.replace('.', 'p').replace('-', '_')
        logging.info(f"Generated run name: {run_name}")
        return run_name


def main():
    """
    Parse arguments, setup configuration, and start training.
    """
    parser = argparse.ArgumentParser(description="Train FairPO Model for Multi-Label Classification")

    home_candidate = Path.home()
    raid_candidate = Path("/raid/speech/user")
    if raid_candidate.exists() and "raid" in str(Path.cwd()).lower():
        user_dir_default = str(raid_candidate)
    elif home_candidate.exists():
        user_dir_default = str(home_candidate)
    else:
        user_dir_default = "."
        logging.warning(f"Defaulting user_dir to '.' for dataset paths.")

    parser.add_argument('--dataset_name', type=str, default='coco', choices=['coco', 'nuswide'])
    parser.add_argument('--coco_root', type=str, default=f"{user_dir_default}/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014")
    parser.add_argument('--nus_root', type=str, default=f"{user_dir_default}/.cache/nus_wide")
    parser.add_argument('--index_dir', type=str, default=None)
    parser.add_argument('--force_regenerate_index', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='./output/ckpt_fairpo_neurips')
    parser.add_argument('--ref_cls_weights_path', type=str, default=None)

    coco_priv_default = "78,70,12,21,76,79,52,54,68,18,31,47,51,10,11,49,64,22,50,19"
    parser.add_argument('--privileged_indices', type=str, default=coco_priv_default)

    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('--train_frac', type=float, default=1.0)
    parser.add_argument('--val_frac', type=float, default=1.0)

    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)

    parser.add_argument('--is_ref_training', action='store_true')
    parser.add_argument('--loss_type', default='dpo', choices=['dpo', 'simpo', 'cpo'])
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--eta_alpha', type=float, default=0.05)
    parser.add_argument('--ema_decay', type=float, default=0.9)
    parser.add_argument('--delta_scaling', type=float, default=1e-6)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--wandb_project', type=str, default="FairPO-NeurIPS")
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--run_name', type=str, default='generate')

    args = parser.parse_args()
    args.is_wandb = not args.no_wandb

    if args.dataset_name == 'nuswide' and args.privileged_indices == coco_priv_default:
        nus_priv_default = ",".join(map(str, range(16)))
        logging.warning(f"Using default privileged indices for NUS-WIDE: '{nus_priv_default}'. Please specify if different.")
        args.privileged_indices = nus_priv_default
    elif args.dataset_name == 'coco' and args.privileged_indices != coco_priv_default:
        logging.info(f"Using custom privileged indices for COCO: {args.privileged_indices}")

    if not args.is_ref_training and args.ref_cls_weights_path is None:
        logging.error("FATAL: --ref_cls_weights_path is REQUIRED when --is_ref_training is False.")
        sys.exit(1)

    trainer = ModelTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
