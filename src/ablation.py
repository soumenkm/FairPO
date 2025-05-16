# ablation.py
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
from typing import Dict, Any, Tuple, Optional, List

from models import VisionModelForCLS # Assuming this is correctly defined
from dataset import COCODatasetOnDemand, NUSWIDEDatasetOnDemand, collate_fn_skip_none # Assuming these are correctly defined
from metrics import Metrics # Assuming this is correctly defined

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class AblationModelTrainer:
    """
    Trainer class specifically adapted for ablation studies on the FairPO framework.
    It allows for disabling specific components like the preference loss,
    non-privileged constraint, or GRPO's adaptive weighting.
    """

    def __init__(self, config, ablation_config: Dict[str, bool]):
        """
        Initialize AblationModelTrainer.

        Args:
            config: Namespace object containing general configuration parameters.
            ablation_config: Dictionary specifying which components to disable for ablation.
                             Expected keys: 'no_preference_loss', 'no_np_constraint', 'no_grpo'.
        """
        self.config = config
        self.ablation_config = ablation_config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
        logging.info(f"Using device: {self.device}")
        self.train_step = 0
        self.val_step = 0
        self.start_epoch = 0

        run_name = self._get_run_name() # Run name now includes ablation info
        self.config.run_name = run_name
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.wandb_project / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints will be saved in: {self.checkpoint_dir}")

        logging.info(f"ABLATION CONFIG: {self.ablation_config}")
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

        dataset_class = COCODatasetOnDemand if config.dataset_name == "coco" else NUSWIDEDatasetOnDemand
        split_name_val = "val" if config.dataset_name == "coco" else "test"

        self.train_dataset = dataset_class(
            root_dir=str(dataset_root), frac=config.train_frac, split_name="train",
            privileged_indices_set=privileged_indices_set, seed=seed,
            index_dir=str(self.config.index_dir), force_regenerate=config.force_regenerate_index
        )
        self.val_dataset = dataset_class(
            root_dir=str(dataset_root), frac=config.val_frac, split_name=split_name_val,
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
        
        # For ablation, we assume FairPO mode (not SFT), so reference weights are always needed.
        if ref_weights_path is None or not ref_weights_path.is_file():
            logging.error(f"Reference classifier weights REQUIRED for ablation studies but not found: '{config.ref_cls_weights_path}'.")
            logging.error("Please provide a valid path to SFT model weights using --ref_cls_weights_path.")
            sys.exit(1)
        logging.info(f"Using reference classifier weights for FairPO (ablation mode): {ref_weights_path}")

        self.model = VisionModelForCLS(
            device=self.device, model_name=config.model_name, num_labels=self.num_labels,
            ref_cls_weights_path=str(ref_weights_path), # Always pass for FairPO based ablations
            privileged_indices=self.privileged_indices,
            non_privileged_indices=self.non_privileged_indices,
            is_ref=False, # Ablations are on FairPO variants, not SFT itself
            loss_type=config.loss_type, # Base loss type for privileged group
            beta=config.beta, epsilon=config.epsilon,
            quant_config=None 
        ).to(self.device)

        self.model.calc_num_params()
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(self.trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

        # GRPO related, conditionally used based on ablation_config['no_grpo']
        self.loss_priv_ema = torch.tensor(0.0, device=self.device)
        self.loss_non_priv_ema = torch.tensor(0.0, device=self.device)
        self.num_updates_priv_loss_ema = 0
        self.num_updates_non_priv_loss_ema = 0
        # Alphas are fixed if GRPO is off
        self.alpha_privileged = torch.tensor(0.5, device=self.device)
        self.alpha_non_privileged = torch.tensor(0.5, device=self.device)
        if ablation_config.get('no_grpo', False):
            logging.info("GRPO disabled for this ablation run. Using fixed alpha weights (0.5/0.5).")
        else:
            logging.info("GRPO enabled for this run (or not specified to be disabled).")

        self._setup_wandb()

    def _calculate_norm(self, model_params, norm_type=2.0):
        total_norm = torch.tensor(0.0, device=self.device)
        for p in model_params:
            if p is not None:
                param_norm = p.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)

    def _calculate_grad_norm(self, model_params, norm_type=2.0):
        total_norm = torch.tensor(0.0, device=self.device)
        params_to_norm = [p for p in model_params if p is not None and p.grad is not None]
        if not params_to_norm:
            return total_norm
        for p in params_to_norm:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)

    def _setup_wandb(self):
        if self.config.is_wandb:
            try:
                wandb.init(
                    project=self.config.wandb_project, # Consider a dedicated ablation project or tags
                    name=self.config.run_name,
                    config={**vars(self.config), "ablation_config": self.ablation_config},
                    tags=["ablation"] + [f"ablate_{k}" for k, v in self.ablation_config.items() if v]
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
                self.config.is_wandb = False # Fallback to no WandB
        else:
            logging.info("WandB disabled.")

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'val_step': self.val_step,
            "config": vars(self.config),
            "ablation_config": self.ablation_config
        }
        # Always include alpha and EMA states, even if GRPO is off (they just won't change)
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

    def _compute_ablated_losses(self, 
                                prob_scores: torch.Tensor, 
                                ref_prob_scores: torch.Tensor, 
                                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes privileged and non-privileged losses according to ablation settings.
        This method directly calls the underlying loss computation functions from Metrics,
        applying ablation logic.
        """
        loss_p = torch.tensor(0.0, device=self.device)
        loss_np = torch.tensor(0.0, device=self.device)

        # Privileged Loss Computation
        if not self.ablation_config.get('no_preference_loss', False):
            # Default to configured loss_type (dpo, simpo, cpo)
            if self.config.loss_type == "dpo":
                loss_p = Metrics._compute_privileged_loss_dpo_conditional(
                    prob_scores, ref_prob_scores, labels, self.privileged_indices,
                    self.num_labels, self.config.beta, self.device
                )
            elif self.config.loss_type == "simpo":
                loss_p = Metrics._compute_privileged_loss_simpo_conditional(
                    prob_scores, labels, self.privileged_indices, self.num_labels,
                    self.config.beta, getattr(self.config, 'gamma_simpo', 0.0), self.device # Add gamma_simpo to config if using
                )
            elif self.config.loss_type == "cpo":
                 loss_p = Metrics._compute_privileged_loss_cpo_conditional(
                    prob_scores, labels, self.privileged_indices, self.num_labels,
                    self.config.beta, getattr(self.config, 'lambda_cpo_nll', 0.5), self.device # Add lambda_cpo_nll
                )
            else:
                raise ValueError(f"Unsupported loss_type for privileged loss: {self.config.loss_type}")
        else: # 'no_preference_loss' is True, so use BCE for privileged labels
            logging.debug("Ablation: Using BCE for privileged loss.")
            if self.privileged_indices: # Only compute if there are privileged indices
                priv_indices_tensor = torch.tensor(self.privileged_indices, device=self.device, dtype=torch.long)
                # Ensure indices are valid before slicing
                valid_priv_indices = priv_indices_tensor[priv_indices_tensor < prob_scores.shape[1]]
                if valid_priv_indices.numel() > 0:
                    scores_priv_subset = prob_scores[:, valid_priv_indices]
                    labels_priv_subset = labels[:, valid_priv_indices]
                    loss_p = Metrics._compute_bce_loss(scores_priv_subset, labels_priv_subset, reduction='mean')
                else:
                    loss_p = torch.tensor(0.0, device=self.device) # No valid privileged indices
            else:
                 loss_p = torch.tensor(0.0, device=self.device) # No privileged indices defined

        # Non-Privileged Loss Computation
        if not self.ablation_config.get('no_np_constraint', False):
            loss_np = Metrics._compute_non_privileged_loss(
                prob_scores, ref_prob_scores, labels, self.non_privileged_indices,
                self.config.epsilon, self.device
            )
        else: # 'no_np_constraint' is True, so use BCE for non-privileged labels
            logging.debug("Ablation: Using BCE for non-privileged loss.")
            if self.non_privileged_indices: # Only compute if there are non-privileged indices
                non_priv_indices_tensor = torch.tensor(self.non_privileged_indices, device=self.device, dtype=torch.long)
                valid_non_priv_indices = non_priv_indices_tensor[non_priv_indices_tensor < prob_scores.shape[1]]
                if valid_non_priv_indices.numel() > 0:
                    scores_non_priv_subset = prob_scores[:, valid_non_priv_indices]
                    labels_non_priv_subset = labels[:, valid_non_priv_indices]
                    loss_np = Metrics._compute_bce_loss(scores_non_priv_subset, labels_non_priv_subset, reduction='mean')
                else:
                    loss_np = torch.tensor(0.0, device=self.device)
            else:
                loss_np = torch.tensor(0.0, device=self.device)
                
        return {"privileged": loss_p, "non_privileged": loss_np}


    def _process_batch_ablation(self, batch: Dict[str, torch.Tensor]) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Process a single batch, applying ablation logic for loss computation.
        The VisionModelForCLS.forward should still return its standard set of metrics.
        The ablation logic here focuses on how the 'privileged' and 'non_privileged'
        components of the loss_components dictionary are derived before GRPO.
        """
        if batch is None or 'pixels' not in batch or batch['pixels'].numel() == 0:
            logging.warning("Skipping empty or invalid batch.")
            return None

        pixels = batch['pixels'].to(self.device)
        labels = batch['labels'].to(self.device)
        current_batch_size = pixels.size(0)

        # Get hidden states and reference scores (ref_classifier is part of self.model)
        with torch.no_grad(): # Backbone is frozen
            vit_outputs = self.model.model.vit(pixel_values=pixels, output_hidden_states=False)
            hidden_state = vit_outputs.last_hidden_state[:, 0, :].to(torch.float32)
        
        current_prob_scores = self.model._get_scores(self.model.model.classifier, hidden_state)
        
        ref_prob_scores = None
        if hasattr(self.model.model, 'ref_classifier'): # Check if ref_classifier exists
            with torch.no_grad():
                ref_prob_scores = self.model._get_scores(self.model.model.ref_classifier, hidden_state)
        else: # Should not happen if ref_weights_path was correctly loaded
            logging.error("Reference classifier not found in model during ablation. Ensure SFT model was loaded.")
            # Fallback or error, for now, let's use current scores as a placeholder if really missing
            # This will make DPO terms zero, and non-priv constraint based on current loss.
            # A better solution is to ensure ref_classifier is always present for FairPO-based ablations.
            ref_prob_scores = current_prob_scores.detach() # Placeholder if missing

        # Compute losses based on ablation settings
        loss_components_ablated = self._compute_ablated_losses(current_prob_scores, ref_prob_scores, labels)

        # Compute standard metrics (accuracy, mAP, etc.) using current_prob_scores
        # These are independent of the loss ablation for evaluation purposes
        accuracy_components = Metrics.compute_accuracy_components(
            current_prob_scores, labels, self.privileged_indices, self.non_privileged_indices
        )
        map_components = Metrics.compute_map_components(
            current_prob_scores, labels, self.privileged_indices, self.non_privileged_indices
        )
        f1_components = Metrics.compute_f1_components(
            current_prob_scores, labels, self.privileged_indices, self.non_privileged_indices
        )
        em_components = Metrics.compute_exact_match_accuracy_components(
            current_prob_scores, labels, self.privileged_indices, self.non_privileged_indices
        )

        output_dict = {
            "outputs": current_prob_scores,
            "loss": loss_components_ablated, # Use the ablated loss components
            "acc": accuracy_components,
            "map": map_components,
            "f1": f1_components,
            "em": em_components,
        }
        return output_dict, current_batch_size


    def _update_epoch_metrics(self, epoch_metrics: Dict[str, Any], results_dict: Dict[str, Any], batch_size: int):
        loss_comp = results_dict.get('loss') # This will be {'privileged': ..., 'non_privileged': ...} from _process_batch_ablation
                                         # and potentially {'combined_loss_for_opt': ...} added later
        if loss_comp:
            priv_loss_val = loss_comp.get('privileged', torch.tensor(0.0)).item()
            non_priv_loss_val = loss_comp.get('non_privileged', torch.tensor(0.0)).item()
            # combined_loss_for_opt is added after GRPO, so it's what's used for optimization
            combined_loss_val = loss_comp.get('combined_loss_for_opt', torch.tensor(0.0)).item()

            epoch_metrics['loss_priv'] += priv_loss_val * batch_size
            epoch_metrics['loss_non_priv'] += non_priv_loss_val * batch_size
            epoch_metrics['loss_total'] += combined_loss_val * batch_size # This tracks the actual optimized loss

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
        loss.backward()
        grad_norm = self._calculate_grad_norm(self.trainable_params)

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.grad_clip)

        self.optimizer.step()
        return grad_norm.item()

    def _log_wandb_train_batch_metrics(self, results_dict: Dict[str, Any], grad_norm: Optional[float], param_norm: float, epoch: int, batch_idx: int, num_batches: int):
        if not self.config.is_wandb: return

        batch_log = {'train/step': self.train_step, 'epoch': epoch + (batch_idx + 1) / num_batches}
        loss_comp = results_dict.get('loss')
        if loss_comp:
            batch_log['train/privileged_loss_batch_avg'] = loss_comp.get('privileged', torch.tensor(np.nan)).item()
            batch_log['train/non_privileged_loss_batch_avg'] = loss_comp.get('non_privileged', torch.tensor(np.nan)).item()
            batch_log['train/combined_loss_for_opt'] = loss_comp.get('combined_loss_for_opt', torch.tensor(np.nan)).item() 
            batch_log["train/alpha_p"] = self.alpha_privileged.item() # Log alphas even if fixed
            batch_log["train/alpha_np"] = self.alpha_non_privileged.item()
            if not self.ablation_config.get('no_grpo', False): # Log EMAs only if GRPO is active
                batch_log["train/loss_priv_ema"] = self.loss_priv_ema.item()
                batch_log["train/loss_non_priv_ema"] = self.loss_non_priv_ema.item()

        # Metrics logging (same as before)
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

        if grad_norm is not None: batch_log['train/grad_norm'] = grad_norm
        batch_log['train/param_norm'] = param_norm
        batch_log['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        wandb.log({k: v for k, v in batch_log.items() if not (isinstance(v, float) and np.isnan(v))})


    def _calculate_epoch_averages(self, epoch_metrics: Dict[str, Any], processed_items: int) -> Dict[str, float]:
        avg_metrics = {}
        if processed_items == 0: 
            logging.warning("No items processed in epoch, metrics will be NaN.")
            # All keys in epoch_metrics are sums, need to be averaged
            for k_template in epoch_metrics.keys():
                avg_metrics[f'avg_{k_template}'] = np.nan
            return avg_metrics

        for key, value in epoch_metrics.items():
            avg_metrics[f'avg_{key}'] = value / processed_items if not np.isnan(value) else np.nan
        return avg_metrics

    def train(self):
        logging.info("Starting ablation training...")
        best_val_metric = float('inf') 
        no_improvement_epochs = 0

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            epoch_metrics_sum = { # For accumulating sums
                'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0, # 'loss_sft' not used in ablation
                'acc_overall': 0.0, 'acc_priv': 0.0, 'acc_non_priv': 0.0,
                'em': 0.0, 'em_priv': 0.0, 'em_non_priv': 0.0,
                'f1': 0.0, 'f1_priv': 0.0, 'f1_non_priv': 0.0,
                'map_overall': 0.0, 'map_priv': 0.0, 'map_non_priv': 0.0
            }
            processed_items_train = 0
            num_batches = len(self.train_loader)

            batch_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} (Ablation)", unit="batch")
            start_time = time.time()

            for i, batch in enumerate(batch_iter):
                if batch is None or 'pixels' not in batch or batch['pixels'].numel() == 0:
                    continue
                current_batch_size = batch['pixels'].size(0)

                # Use the ablated batch processing
                processed_batch_info = self._process_batch_ablation(batch)
                if processed_batch_info is None: continue
                
                results_dict, _ = processed_batch_info # results_dict['loss'] has {'privileged': ..., 'non_privileged': ...}
                loss_components_ablated = results_dict['loss']
                self.optimizer.zero_grad()

                # These are per-batch average losses for groups based on ablation settings
                loss_priv_batch_avg = loss_components_ablated.get("privileged", torch.tensor(0.0, device=self.device))
                loss_non_priv_batch_avg = loss_components_ablated.get("non_privileged", torch.tensor(0.0, device=self.device))
                
                # GRPO or fixed alpha weighting
                if not self.ablation_config.get('no_grpo', False): # GRPO is active
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
                # else: Alphas remain fixed at 0.5/0.5 (or whatever they were initialized to)
            
                combined_loss_for_opt = self.alpha_privileged * loss_priv_batch_avg + \
                                        self.alpha_non_privileged * loss_non_priv_batch_avg
                results_dict['loss']['combined_loss_for_opt'] = combined_loss_for_opt # Store for logging
                    
                grad_norm_val = self._perform_optimization_step(combined_loss_for_opt)
        
                self.train_step += 1
                param_norm_val = self._calculate_norm(self.trainable_params).item()

                self._update_epoch_metrics(epoch_metrics_sum, results_dict, current_batch_size)
                processed_items_train += current_batch_size

                if self.config.is_wandb:
                    self._log_wandb_train_batch_metrics(results_dict, grad_norm_val, param_norm_val, epoch, i, num_batches)

                tqdm_postfix = {
                    'loss': combined_loss_for_opt.item(),
                    'acc': results_dict.get('acc', {}).get('acc', np.nan),
                    'step': self.train_step,
                    'alpha_p': self.alpha_privileged.item() # Always show alpha
                }
                batch_iter.set_postfix(**{k: f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else (str(v) if isinstance(v, float) and np.isnan(v) else v) for k,v in tqdm_postfix.items()})

            epoch_duration = time.time() - start_time
            logging.info(f"Epoch {epoch+1} training completed in {epoch_duration:.2f}s")

            avg_train_metrics = self._calculate_epoch_averages(epoch_metrics_sum, processed_items_train)
            avg_train_metrics_wandb = {f"train/{k.replace('avg_', '')}": v for k, v in avg_train_metrics.items()} 
            
            if self.config.is_wandb:
                wandb.log({**avg_train_metrics_wandb, 'epoch': epoch + 1, 'train/step': self.train_step})

            avg_val_metrics_dict_prefixed = self._validate(epoch)
            
            # Metric for early stopping and best model (consistent for ablations)
            # Using negative privileged mAP (higher is better, so lower negative is better)
            current_val_metric_for_best = -avg_val_metrics_dict_prefixed.get('val/map_priv', float('-inf'))
            is_best = current_val_metric_for_best < best_val_metric

            if is_best:
                best_val_metric = current_val_metric_for_best
                logging.info(f"New best validation metric achieved: {best_val_metric:.4f} (based on -mAP_priv)")
                self._save_checkpoint(epoch, is_best=True)
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                logging.info(f"Validation metric (-mAP_priv) did not improve. Current: {current_val_metric_for_best:.4f}, Best: {best_val_metric:.4f}. Streak: {no_improvement_epochs}")

            self._save_checkpoint(epoch, is_best=False)
            
            self._log_epoch_summary_console(epoch, avg_train_metrics, mode='train')
            self._log_epoch_summary_console(epoch, avg_val_metrics_dict_prefixed, mode='val')

            if self.config.early_stopping_patience > 0 and no_improvement_epochs >= self.config.early_stopping_patience:
                logging.info(f"Early stopping triggered after {no_improvement_epochs} epochs.")
                break

        logging.info("Ablation training finished.")
        if self.config.is_wandb:
            logging.info("Finishing WandB run.")
            wandb.summary['best_val_metric_achieved'] = best_val_metric # (-mAP_priv)
            wandb.finish()

    def _validate(self, epoch: int) -> Dict[str, float]:
        logging.info(f"Starting validation for epoch {epoch+1} (Ablation)...")
        self.model.eval()
        val_epoch_metrics_sum = {
            'loss_priv': 0.0, 'loss_non_priv': 0.0, 'loss_total': 0.0,
            'acc_overall': 0.0, 'acc_priv': 0.0, 'acc_non_priv': 0.0,
            'em': 0.0, 'em_priv': 0.0, 'em_non_priv': 0.0,
            'f1': 0.0, 'f1_priv': 0.0, 'f1_non_priv': 0.0,
            'map_overall': 0.0, 'map_priv': 0.0, 'map_non_priv': 0.0
        }
        processed_items_val = 0

        with torch.no_grad():
            val_iter = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1} (Ablation)", unit="batch", leave=False)
            for batch in val_iter:
                if batch is None or 'pixels' not in batch or batch['pixels'].numel() == 0:
                    continue
                current_batch_size = batch['pixels'].size(0)
                
                processed_batch_info = self._process_batch_ablation(batch) # Use ablated processing
                if processed_batch_info is None: continue
                results_dict, _ = processed_batch_info

                # For validation, calculate combined loss using current alphas for consistency
                # This is important for comparing across epochs/runs fairly
                loss_priv_val = results_dict['loss'].get('privileged', torch.tensor(0.0, device=self.device))
                loss_non_priv_val = results_dict['loss'].get('non_privileged', torch.tensor(0.0, device=self.device))
                results_dict['loss']['combined_loss_for_opt'] = self.alpha_privileged * loss_priv_val + self.alpha_non_privileged * loss_non_priv_val

                self._update_epoch_metrics(val_epoch_metrics_sum, results_dict, current_batch_size)
                processed_items_val += current_batch_size

        avg_val_metrics_raw = self._calculate_epoch_averages(val_epoch_metrics_sum, processed_items_val)
        avg_val_metrics_prefixed = {f"val/{k.replace('avg_', '')}": v for k, v in avg_val_metrics_raw.items()}

        self.val_step += 1
        if self.config.is_wandb:
            self._log_wandb_validation_summary(epoch, avg_val_metrics_prefixed)

        self.model.train() 
        return avg_val_metrics_prefixed

    def _log_epoch_summary_console(self, epoch: int, avg_epoch_metrics_dict: Dict[str, float], mode: str):
        is_val_mode = mode == 'val'
        key_prefix = "val/" if is_val_mode else "avg_"
        title_prefix = "Training" if mode == 'train' else "Validation"
        step_info = f"(Train Step {self.train_step})" if mode == 'train' else f"(Val Step {self.val_step})"
        logging.info(f"--- {title_prefix} Epoch {epoch+1} Results {step_info} (Ablation) ---")
        logging.info(f"  Ablation Config: {self.ablation_config}") # Remind user of ablation settings

        # Generic FairPO style logging (no SFT loss here)
        logging.info(f"  Avg Loss (Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}loss_priv', np.nan):.4f}")
        logging.info(f"  Avg Loss (Non-Priv): {avg_epoch_metrics_dict.get(f'{key_prefix}loss_non_priv', np.nan):.4f}")
        logging.info(f"  Avg Loss (Combined): {avg_epoch_metrics_dict.get(f'{key_prefix}loss_total', np.nan):.4f}")
        if mode == 'train': 
            logging.info(f"  Alphas (P/NP): {self.alpha_privileged.item():.3f} / {self.alpha_non_privileged.item():.3f}")
            if not self.ablation_config.get('no_grpo', False):
                logging.info(f"  Loss EMAs (P/NP): {self.loss_priv_ema.item():.4f} / {self.loss_non_priv_ema.item():.4f}")
        
        # Standard metrics
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
        if not self.config.is_wandb: return
        log_payload = {**val_metrics_prefixed}
        log_payload['train/step'] = self.train_step 
        log_payload['val/step'] = self.val_step 
        log_payload['epoch'] = epoch + 1 
        wandb.log({k: v for k, v in log_payload.items() if not (isinstance(v, float) and np.isnan(v))})
        
    def _get_run_name(self) -> str:
        base_run_name = self.config.run_name
        if base_run_name and base_run_name != 'generate':
            # Append ablation info to user-provided name
            ablation_suffix = "_ablation"
            if self.ablation_config.get('no_preference_loss', False): ablation_suffix += "_noPref"
            if self.ablation_config.get('no_np_constraint', False): ablation_suffix += "_noNPConst"
            if self.ablation_config.get('no_grpo', False): ablation_suffix += "_noGRPO"
            return f"{base_run_name}{ablation_suffix}"

        # Auto-generate name
        dataset_prefix = self.config.dataset_name.upper()
        # For ablations, mode is always FairPO-like, but we indicate ablation
        mode = "FairPO-Ablation"
        loss_part = f"_{self.config.loss_type}" # Base loss type before ablation
        
        ablation_tags = []
        if self.ablation_config.get('no_preference_loss', False): ablation_tags.append("noPref")
        if self.ablation_config.get('no_np_constraint', False): ablation_tags.append("noNPConst")
        if self.ablation_config.get('no_grpo', False): ablation_tags.append("noGRPO")
        ablation_str = ("_" + "-".join(ablation_tags)) if ablation_tags else ""

        lr_str = f"lr{self.config.learning_rate:.0e}"
        ep_str = f"ep{self.config.epochs}"
        
        # FairPO params that are still relevant even in ablation (e.g., beta for CPO/DPO if pref loss is on)
        # Eta_alpha is only relevant if GRPO is on. Epsilon only if NP constraint is on.
        # For simplicity in generated name, just include beta if preference loss is ON, and epsilon if NP constraint is ON
        # The full ablation_config is logged to wandb anyway.
        
        active_fairpo_params = []
        if not self.ablation_config.get('no_preference_loss', False):
            active_fairpo_params.append(f"beta{self.config.beta}")
        if not self.ablation_config.get('no_np_constraint', False):
            active_fairpo_params.append(f"eps{self.config.epsilon}")
        if not self.ablation_config.get('no_grpo', False): # GRPO related params only if GRPO is ON
            active_fairpo_params.append(f"eta{self.config.eta_alpha}")
            active_fairpo_params.append(f"ema{self.config.ema_decay}")
            active_fairpo_params.append(f"deltaS{self.config.delta_scaling}")
        
        fairpo_params_str = ("_" + "_".join(active_fairpo_params) + "_") if active_fairpo_params else "_"

        run_name = f"{dataset_prefix}_{mode}{loss_part}{ablation_str}_{lr_str}{fairpo_params_str}{ep_str}"
        run_name = run_name.replace('.', 'p').replace('-', '_').rstrip('_') # Sanitize and remove trailing underscore
        logging.info(f"Generated run name (ablation): {run_name}")
        return run_name

def run_single_ablation(base_config_args: argparse.Namespace, 
                        ablation_setting: Dict[str, bool], 
                        ablation_name_suffix: str):
    """
    Runs a single ablation experiment with the given base configuration and ablation setting.
    """
    # Create a copy of base_config_args to modify for this run if needed, or pass directly
    # For ablation, the main config arguments (lr, epochs, dataset etc) are usually fixed from base_config_args
    
    logging.info(f"\n{'='*20} STARTING ABLATION: {ablation_name_suffix} {'='*20}")
    logging.info(f"Ablation setting: {ablation_setting}")

    # Modify run_name or let the trainer generate it based on ablation_setting
    # If base_config_args.run_name was 'generate', the AblationModelTrainer will append ablation info.
    # If it was specific, AblationModelTrainer will also append.

    trainer = AblationModelTrainer(base_config_args, ablation_setting)
    trainer.train()
    logging.info(f"\n{'='*20} FINISHED ABLATION: {ablation_name_suffix} {'='*20}\n")


def main_ablation():
    """
    Main function to define and run ablation studies.
    Parses base arguments and then iterates through different ablation configurations.
    """
    parser = argparse.ArgumentParser(description="Run Ablation Studies for FairPO Model")

    # --- Base Configuration Arguments (same as train.py) ---
    home_candidate = Path.home()
    raid_candidate = Path("/raid/speech/user")
    user_dir_default = str(raid_candidate) if raid_candidate.exists() and "raid" in str(Path.cwd()).lower() else str(home_candidate) if home_candidate.exists() else "."

    parser.add_argument('--dataset_name', type=str, default='coco', choices=['coco', 'nuswide'])
    parser.add_argument('--coco_root', type=str, default=f"{user_dir_default}/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014")
    parser.add_argument('--nus_root', type=str, default=f"{user_dir_default}/.cache/nus_wide")
    parser.add_argument('--index_dir', type=str, default=None)
    parser.add_argument('--force_regenerate_index', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='./output/ckpt_fairpo_ablations_neurips') # Dedicated dir for ablations
    parser.add_argument('--ref_cls_weights_path', type=str, required=True, help='Path to SFT pre-trained reference model weights (MANDATORY for ablations).')

    coco_priv_default = "78,70,12,21,76,79,52,54,68,18,31,47,51,10,11,49,64,22,50,19"
    parser.add_argument('--privileged_indices', type=str, default=coco_priv_default)
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('--train_frac', type=float, default=1.0)
    parser.add_argument('--val_frac', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=25) # Use same epochs as full model for fair comparison
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=5) # Keep early stopping
    
    # FairPO specific params (these are the base for the full model, ablations modify behavior)
    # is_ref_training is implicitly False for ablations (we ablate FairPO components)
    parser.add_argument('--loss_type', default='cpo', choices=['dpo', 'simpo', 'cpo'], help='Base conditional loss for privileged group (before ablation). CPO is often best.')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--eta_alpha', type=float, default=0.05)
    parser.add_argument('--ema_decay', type=float, default=0.9)
    parser.add_argument('--delta_scaling', type=float, default=1e-6)
    # parser.add_argument('--gamma_simpo', type=float, default=0.0) # If testing SimPO
    # parser.add_argument('--lambda_cpo_nll', type=float, default=0.5) # If testing CPO

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--wandb_project', type=str, default="FairPO-Ablations-NeurIPS") # Dedicated project
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--run_name_base', type=str, default='generate', help='Base for run name (ablation suffixes will be added). Default "generate" to auto-generate full name.')

    # Ablation control arguments (NEW)
    parser.add_argument('--run_all_ablations', action='store_true', help='If set, run all predefined ablation configurations sequentially.')
    parser.add_argument('--ablation_no_preference_loss', action='store_true', help='Ablate preference loss for privileged (use BCE).')
    parser.add_argument('--ablation_no_np_constraint', action='store_true', help='Ablate non-privileged constraint (use BCE).')
    parser.add_argument('--ablation_no_grpo', action='store_true', help='Ablate GRPO (use fixed 0.5/0.5 weights).')
    # Add more specific ablations if needed, e.g., --ablation_only_conf_negatives (would require modifying _compute_ablated_losses or Metrics)

    args = parser.parse_args()
    args.is_wandb = not args.no_wandb

    if args.dataset_name == 'nuswide' and args.privileged_indices == coco_priv_default:
        nus_priv_default = ",".join(map(str, range(16))) 
        logging.warning(f"Using default privileged indices for NUS-WIDE: '{nus_priv_default}'.")
        args.privileged_indices = nus_priv_default
    
    # --- Define Ablation Configurations ---
    # Full model (no ablations, for reference within ablation runs if desired, or run separately with train.py)
    # This script focuses on running the *ablated* versions.
    # The "full FairPO-CPO" would be run by train.py with --loss_type cpo and no ablation flags.
    
    ablation_configurations = []
    if args.run_all_ablations:
        ablation_configurations.extend([
            ({"no_preference_loss": True, "no_np_constraint": False, "no_grpo": False}, "NoPrefLoss"),
            ({"no_preference_loss": False, "no_np_constraint": True, "no_grpo": False}, "NoNPConstraint"),
            ({"no_preference_loss": False, "no_np_constraint": False, "no_grpo": True}, "NoGRPO"),
            # Example of combined ablation:
            # ({"no_preference_loss": True, "no_np_constraint": True, "no_grpo": False}, "NoPref_NoNPConst"),
        ])
    else:
        # Run a single ablation based on command-line flags
        current_ablation_setting = {
            "no_preference_loss": args.ablation_no_preference_loss,
            "no_np_constraint": args.ablation_no_np_constraint,
            "no_grpo": args.ablation_no_grpo
        }
        # Check if any ablation flag is set to actually run an ablation
        if any(current_ablation_setting.values()):
            suffix_parts = []
            if args.ablation_no_preference_loss: suffix_parts.append("NoPrefLoss")
            if args.ablation_no_np_constraint: suffix_parts.append("NoNPConstraint")
            if args.ablation_no_grpo: suffix_parts.append("NoGRPO")
            ablation_suffix = "_".join(suffix_parts) if suffix_parts else "CustomAblation"
            ablation_configurations.append((current_ablation_setting, ablation_suffix))
        else:
            logging.info("No specific ablation flags set and --run_all_ablations not used. Exiting. "
                         "To run the full model, use train.py. "
                         "To run specific ablations, use --ablation_... flags or --run_all_ablations.")
            return

    if not ablation_configurations:
        logging.warning("No ablation configurations to run. Exiting.")
        return

    for i, (setting, suffix) in enumerate(ablation_configurations):
        # Create a fresh copy of args for each run to avoid state leakage if args were modified
        current_run_args = argparse.Namespace(**vars(args)) 
        
        # Set the run_name for this specific ablation run (AblationModelTrainer will use this)
        if args.run_name_base == 'generate':
            current_run_args.run_name = 'generate' # Let AblationModelTrainer fully generate it
        else:
            current_run_args.run_name = f"{args.run_name_base}_{suffix}" # Append suffix to user's base

        run_single_ablation(current_run_args, setting, suffix)
        
        if i < len(ablation_configurations) - 1:
            logging.info(f"Pausing for a few seconds before next ablation run...")
            time.sleep(5) # Small pause, optional

if __name__ == '__main__':
    # Example: CUDA_VISIBLE_DEVICES=1 python ablation.py --dataset_name coco --ref_cls_weights_path <path_to_sft_coco.pth> --run_all_ablations
    # Example: CUDA_VISIBLE_DEVICES=1 python ablation.py --dataset_name coco --ref_cls_weights_path <path_to_sft_coco.pth> --ablation_no_grpo --loss_type cpo
    main_ablation()