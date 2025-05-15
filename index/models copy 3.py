import os
import sys
from pathlib import Path
from typing import Union, List, Set, Optional, Dict, Any
import copy
import logging

import torchinfo # Keep if you use it, otherwise can be removed
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModel,
                          BitsAndBytesConfig)
import torchmetrics # Import torchmetrics

# Metrics class import can be removed if calculate_std_metrics is self-contained
# from metrics import Metrics 

seed = 42
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class VisionModelForCLS(torch.nn.Module):
    def __init__(self,
                 device: torch.device,
                 model_name: str,
                 num_labels: int,
                 ref_cls_weights_path: Optional[str],
                 privileged_indices: Set[int],
                 non_privileged_indices: Set[int],
                 is_ref: bool,
                 loss_type: Optional[str],
                 beta: float = 1.0,
                 epsilon: float = 0.1,
                 quant_config: Union[BitsAndBytesConfig, None] = None):
        super(VisionModelForCLS, self).__init__()
        self.device = device
        self.model_name = model_name
        self.quant_config = quant_config
        self.num_labels = num_labels
        self.ref_cls_wt_path = ref_cls_weights_path
        
        self.privileged_indices_set = privileged_indices
        self.non_privileged_indices_set = non_privileged_indices
        
        self.is_ref = is_ref
        self.beta = beta
        self.epsilon = epsilon
        self.eps_log = 1e-8 
        self.loss_type = loss_type

        self.label_is_privileged_mask = torch.zeros(self.num_labels, dtype=torch.bool, device=self.device)
        if self.privileged_indices_set:
            self.label_is_privileged_mask[list(self.privileged_indices_set)] = True

        self.vit = AutoModel.from_pretrained(
            self.model_name,
            quantization_config=self.quant_config,
        ).to(self.device)
        
        self.d = self.vit.config.hidden_size
        self.c = num_labels
        self.config = self.vit.config
        
        self.classifier_head = nn.Linear(self.d, self.c).to(self.device)
        
        self.ref_classifier_head = None
        if not self.is_ref:
            self.ref_classifier_head = nn.Linear(self.d, self.c).to(self.device)
            if self.ref_cls_wt_path is not None:
                try:
                    ref_checkpoint = torch.load(self.ref_cls_wt_path, map_location=self.device)
                    ref_head_state_dict = {}
                    for k, v in ref_checkpoint["model_state_dict"].items():
                        if k.startswith("classifier_head."):
                             ref_head_state_dict[k.replace("classifier_head.", "")] = v
                    
                    if not ref_head_state_dict:
                        logging.warning(f"No keys matching 'classifier_head.*' found in reference checkpoint at {self.ref_cls_wt_path}. Attempting to load from 'model.classifier.*' if it was a HF model.")
                        for k, v in ref_checkpoint["model_state_dict"].items():
                            if k.startswith("model.classifier."):
                                ref_head_state_dict[k.replace("model.classifier.", "")] = v

                    if ref_head_state_dict:
                        self.ref_classifier_head.load_state_dict(ref_head_state_dict, strict=True)
                        logging.info(f"Reference classifier weights loaded from {self.ref_cls_wt_path}")
                    else:
                        logging.error(f"Could not find classifier head weights in reference checkpoint: {self.ref_cls_wt_path}. Reference head will be randomly initialized.")

                except Exception as e:
                    logging.error(f"Failed to load or parse reference classifier weights from {self.ref_cls_wt_path}: {e}. Reference head will be randomly initialized.")
            else:
                logging.warning("Reference classifier weights path NOT provided. Reference head will be randomly initialized.")
            
            if self.ref_classifier_head:
                for param in self.ref_classifier_head.parameters():
                    param.requires_grad = False
                self.ref_classifier_head.eval()

        for param in self.vit.parameters():
            param.requires_grad = False
        
        for param in self.classifier_head.parameters():
            param.requires_grad = True

        # --- Initialize TorchMetrics mAP objects ---
        # These will be used for batch-wise mAP. If epoch-wise is needed,
        # they should be managed in the Trainer.
        self.batch_overall_map_metric = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=self.num_labels, average='macro'
        ).to(self.device)
        self.batch_per_class_ap_metric = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=self.num_labels, average=None # Get AP for each class
        ).to(self.device)


    def calc_num_params(self) -> None:
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(self)
        print(f"Backbone: {self.model_name}")
        print(f"Number of total parameters: {total_params:,}")
        print(f"Number of trainable parameters: {train_params:,}")
        if total_params > 0:
           print(f"Training Percentage: {train_params * 100 / total_params:.3f}%")
        else:
           print("Training Percentage: N/A (total_params is zero)")
        
    def forward(self, pixels: torch.Tensor, labels: torch.Tensor,
                sampled_r_indices_for_batch: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        
        batch_size = pixels.size(0)
        
        vit_outputs = self.vit(pixel_values=pixels.to(self.device))
        embeddings = vit_outputs.last_hidden_state[:, 0, :] 
        embeddings = embeddings.to(torch.float32)

        logits_policy = self.classifier_head(embeddings)
        
        logits_ref = None
        if self.ref_classifier_head:
            with torch.no_grad():
                logits_ref = self.ref_classifier_head(embeddings)

        loss_dict_components = {'privileged': torch.tensor(0.0, device=self.device), 
                                'non_privileged': torch.tensor(0.0, device=self.device)}
        
        if self.is_ref:
            sft_loss = nn.BCEWithLogitsLoss()(logits_policy, labels.float().to(self.device))
            loss_dict_components['loss'] = sft_loss
        
        else: 
            if sampled_r_indices_for_batch is None:
                logging.error("sampled_r_indices_for_batch is None during FairPO training! This should not happen.")
            else:
                actual_privileged_losses_this_batch = []
                actual_non_privileged_losses_this_batch = []
                current_labels_device = labels.to(self.device)

                for i in range(batch_size):
                    r_idx = sampled_r_indices_for_batch[i].item()
                    is_r_privileged = self.label_is_privileged_mask[r_idx].item()
                    current_sample_labels = current_labels_device[i]
                    current_sample_logits_policy = logits_policy[i]
                    current_sample_logits_ref = logits_ref[i] if logits_ref is not None else None

                    if is_r_privileged:
                        l_idx = r_idx
                        gt_label_l = current_sample_labels[l_idx]
                        policy_logit_l = current_sample_logits_policy[l_idx]
                        ref_logit_l = torch.tensor(0.0, device=self.device)
                        if current_sample_logits_ref is not None:
                            ref_logit_l = current_sample_logits_ref[l_idx]
                        loss_for_this_priv_sample = torch.tensor(0.0, device=self.device)
                        confusing_set_found_and_processed = False
                        prob_policy_l = torch.sigmoid(policy_logit_l)
                        prob_ref_l = torch.sigmoid(ref_logit_l)

                        if gt_label_l == 1:
                            confusing_negatives_k_indices = []
                            for k_candidate_idx in range(self.num_labels):
                                if k_candidate_idx == l_idx: continue
                                gt_label_k_candidate = current_sample_labels[k_candidate_idx]
                                policy_logit_k_candidate = current_sample_logits_policy[k_candidate_idx]
                                prob_policy_k_candidate = torch.sigmoid(policy_logit_k_candidate)
                                if gt_label_k_candidate == 0 and prob_policy_k_candidate >= prob_policy_l:
                                    confusing_negatives_k_indices.append(k_candidate_idx)
                            if confusing_negatives_k_indices:
                                confusing_set_found_and_processed = True
                                sampled_k_idx = np.random.choice(confusing_negatives_k_indices)
                                policy_logit_k = current_sample_logits_policy[sampled_k_idx]
                                ref_logit_k = current_sample_logits_ref[sampled_k_idx] if current_sample_logits_ref is not None else torch.tensor(0.0, device=self.device)
                                prob_policy_k = torch.sigmoid(policy_logit_k)
                                prob_ref_k = torch.sigmoid(ref_logit_k)
                                log_term_policy_l_vs_ref = torch.log(prob_policy_l + self.eps_log) - torch.log(prob_ref_l + self.eps_log)
                                log_term_policy_k_vs_ref = torch.log(prob_policy_k + self.eps_log) - torch.log(prob_ref_k + self.eps_log)
                                h_val = log_term_policy_l_vs_ref - log_term_policy_k_vs_ref
                                loss_for_this_priv_sample = -torch.log(torch.sigmoid(self.beta * h_val) + self.eps_log)
                        elif gt_label_l == 0:
                            confusing_positives_k_prime_indices = []
                            for k_prime_candidate_idx in range(self.num_labels):
                                if k_prime_candidate_idx == l_idx: continue
                                gt_label_k_prime_candidate = current_sample_labels[k_prime_candidate_idx]
                                policy_logit_k_prime_candidate = current_sample_logits_policy[k_prime_candidate_idx]
                                prob_policy_k_prime_candidate = torch.sigmoid(policy_logit_k_prime_candidate)
                                if gt_label_k_prime_candidate == 1 and prob_policy_k_prime_candidate <= prob_policy_l:
                                    confusing_positives_k_prime_indices.append(k_prime_candidate_idx)
                            if confusing_positives_k_prime_indices:
                                confusing_set_found_and_processed = True
                                sampled_k_prime_idx = np.random.choice(confusing_positives_k_prime_indices)
                                policy_logit_k_prime = current_sample_logits_policy[sampled_k_prime_idx]
                                ref_logit_k_prime = current_sample_logits_ref[sampled_k_prime_idx] if current_sample_logits_ref is not None else torch.tensor(0.0, device=self.device)
                                prob_policy_k_prime = torch.sigmoid(policy_logit_k_prime)
                                prob_ref_k_prime = torch.sigmoid(ref_logit_k_prime)
                                log_term_policy_k_prime_vs_ref = torch.log(prob_policy_k_prime + self.eps_log) - torch.log(prob_ref_k_prime + self.eps_log)
                                log_term_policy_l_vs_ref = torch.log(prob_policy_l + self.eps_log) - torch.log(prob_ref_l + self.eps_log)
                                h_val = log_term_policy_k_prime_vs_ref - log_term_policy_l_vs_ref
                                loss_for_this_priv_sample = -torch.log(torch.sigmoid(self.beta * h_val) + self.eps_log)
                        if not confusing_set_found_and_processed:
                            bce_loss_fn_single = torch.nn.BCEWithLogitsLoss()
                            loss_for_this_priv_sample = bce_loss_fn_single(policy_logit_l.unsqueeze(0), gt_label_l.float().unsqueeze(0))
                        actual_privileged_losses_this_batch.append(loss_for_this_priv_sample)
                    else:
                        j_idx = r_idx
                        gt_label_j = current_sample_labels[j_idx]
                        policy_logit_j = current_sample_logits_policy[j_idx]
                        ref_logit_j = current_sample_logits_ref[j_idx] if current_sample_logits_ref is not None else torch.tensor(0.0, device=self.device)
                        bce_loss_fn_single = torch.nn.BCEWithLogitsLoss()
                        loss_policy_j = bce_loss_fn_single(policy_logit_j.unsqueeze(0), gt_label_j.float().unsqueeze(0))
                        loss_ref_j = bce_loss_fn_single(ref_logit_j.unsqueeze(0), gt_label_j.float().unsqueeze(0))
                        loss_val_non_priv = torch.relu(loss_policy_j - loss_ref_j.detach() - self.epsilon)
                        actual_non_privileged_losses_this_batch.append(loss_val_non_priv)

                if actual_privileged_losses_this_batch:
                    loss_dict_components['privileged'] = torch.mean(torch.stack(actual_privileged_losses_this_batch))
                if actual_non_privileged_losses_this_batch:
                    loss_dict_components['non_privileged'] = torch.mean(torch.stack(actual_non_privileged_losses_this_batch))
        
        metrics_output = self.calculate_std_metrics(logits_policy, labels.to(self.device))

        return {
            "outputs": torch.sigmoid(logits_policy),
            "logits": logits_policy,
            "loss": loss_dict_components,
            "acc": metrics_output['acc'],
            "f1": metrics_output['f1'],
            "map": metrics_output['map'],
            "em": metrics_output['em']
        }

    def calculate_std_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        labels_int = labels.to(logits.device).int() # torchmetrics often expects int labels
        prob_scores = torch.sigmoid(logits)
        preds_float = (prob_scores > 0.5).float() # For acc, em

        # --- Accuracy and Exact Match ---
        overall_acc = (preds_float == labels).float().mean().item()
        exact_match = (torch.all(preds_float == labels, dim=1)).float().mean().item()

        priv_indices_list = list(self.privileged_indices_set) if self.privileged_indices_set else []
        non_priv_indices_list = list(self.non_privileged_indices_set) if self.non_privileged_indices_set else []

        acc_privileged_val = torch.tensor(np.nan, device=logits.device)
        em_privileged_val = torch.tensor(np.nan, device=logits.device)
        if priv_indices_list and preds_float[:, priv_indices_list].numel() > 0:
            acc_privileged_val = (preds_float[:, priv_indices_list] == labels[:, priv_indices_list]).float().mean()
            em_privileged_val = (torch.all(preds_float[:, priv_indices_list] == labels[:, priv_indices_list], dim=1)).float().mean()

        acc_non_privileged_val = torch.tensor(np.nan, device=logits.device)
        em_non_privileged_val = torch.tensor(np.nan, device=logits.device)
        if non_priv_indices_list and preds_float[:, non_priv_indices_list].numel() > 0:
            acc_non_privileged_val = (preds_float[:, non_priv_indices_list] == labels[:, non_priv_indices_list]).float().mean()
            em_non_privileged_val = (torch.all(preds_float[:, non_priv_indices_list] == labels[:, non_priv_indices_list], dim=1)).float().mean()

        # --- mAP Calculation ---
        # Update, compute, and reset for batch-wise mAP
        self.batch_overall_map_metric.update(prob_scores, labels_int)
        map_overall_val = self.batch_overall_map_metric.compute()
        self.batch_overall_map_metric.reset()

        self.batch_per_class_ap_metric.update(prob_scores, labels_int)
        per_class_aps = self.batch_per_class_ap_metric.compute() # Tensor of APs per class
        self.batch_per_class_ap_metric.reset()

        map_privileged_val = torch.tensor(np.nan, device=logits.device)
        if priv_indices_list and per_class_aps is not None and per_class_aps.numel() == self.num_labels:
            # Ensure per_class_aps is not empty and has correct number of elements
            aps_for_privileged = per_class_aps[priv_indices_list]
            valid_priv_aps = aps_for_privileged[~torch.isnan(aps_for_privileged)]
            if valid_priv_aps.numel() > 0:
                map_privileged_val = valid_priv_aps.mean()
        
        map_non_privileged_val = torch.tensor(np.nan, device=logits.device)
        if non_priv_indices_list and per_class_aps is not None and per_class_aps.numel() == self.num_labels:
            aps_for_non_privileged = per_class_aps[non_priv_indices_list]
            valid_non_priv_aps = aps_for_non_privileged[~torch.isnan(aps_for_non_privileged)]
            if valid_non_priv_aps.numel() > 0:
                map_non_privileged_val = valid_non_priv_aps.mean()

        # --- F1 Score (Placeholder - Implement with torchmetrics.F1Score if needed) ---
        f1_overall_val = torch.tensor(np.nan, device=logits.device)
        f1_privileged_val = torch.tensor(np.nan, device=logits.device)
        f1_non_privileged_val = torch.tensor(np.nan, device=logits.device)
        # Example for F1 (you'd need to initialize self.batch_f1_metric etc. like mAP):
        # self.batch_f1_metric.update(prob_scores, labels_int)
        # f1_overall_val = self.batch_f1_metric.compute()
        # self.batch_f1_metric.reset()
        # Similar logic for privileged/non-privileged F1 if `average=None` is used for F1 per class.

        def to_loggable(val_tensor):
            # Handles scalar tensors, converting to Python float or np.nan
            if isinstance(val_tensor, torch.Tensor) and val_tensor.numel() == 1:
                item = val_tensor.item()
                return np.nan if np.isnan(item) else item
            elif isinstance(val_tensor, float) and np.isnan(val_tensor): # Already a Python float nan
                return np.nan
            return val_tensor # Should be a Python float if not np.nan

        return {
            'acc': {'acc': to_loggable(overall_acc), 
                    'privileged': to_loggable(acc_privileged_val), 
                    'non_privileged': to_loggable(acc_non_privileged_val)},
            'em': {'em': to_loggable(exact_match), 
                   'em_privileged': to_loggable(em_privileged_val), 
                   'em_non_privileged': to_loggable(em_non_privileged_val)},
            'f1': {'f1': to_loggable(f1_overall_val), 
                   'f1_privileged': to_loggable(f1_privileged_val), 
                   'f1_non_privileged': to_loggable(f1_non_privileged_val)},
            'map': {'mAP': to_loggable(map_overall_val), 
                    'mAP_privileged': to_loggable(map_privileged_val), 
                    'mAP_non_privileged': to_loggable(map_non_privileged_val)}
        }
        
def main_cls(model_name: str, device: torch.device) -> None:
    logging.basicConfig(level=logging.INFO)
    quant_config = None
    num_classes = 80
    ALL_LABELS_INDICES = list(range(80))
    PRIVILEGED_INDICES_SET = set(ALL_LABELS_INDICES[:20]) 
    NON_PRIVILEGED_INDICES_SET = set(ALL_LABELS_INDICES) - PRIVILEGED_INDICES_SET

    print(f"Privileged Indices: {sorted(list(PRIVILEGED_INDICES_SET))}")
    print(f"Non-Privileged Indices: {sorted(list(NON_PRIVILEGED_INDICES_SET))}")

    print("\n--- Initializing FairPO Model (is_ref=False) ---")
    sft_model_for_ref = VisionModelForCLS(
        device=device, model_name=model_name, num_labels=num_classes,
        ref_cls_weights_path=None, privileged_indices=set(), non_privileged_indices=set(),
        is_ref=True, loss_type=None 
    ).to(device)
    dummy_sft_checkpoint_path = "dummy_sft_ref_checkpoint.pth"
    torch.save({'model_state_dict': sft_model_for_ref.state_dict()}, dummy_sft_checkpoint_path)
    print(f"Saved dummy SFT checkpoint for reference: {dummy_sft_checkpoint_path}")

    fairpo_model = VisionModelForCLS(
        device=device, model_name=model_name, num_labels=num_classes,
        ref_cls_weights_path=dummy_sft_checkpoint_path, 
        privileged_indices=PRIVILEGED_INDICES_SET, non_privileged_indices=NON_PRIVILEGED_INDICES_SET,
        loss_type="dpo", beta=2.0, epsilon=0.01, is_ref=False, quant_config=quant_config
    ).to(device)
    fairpo_model.calc_num_params()

    print("\n--- Initializing SFT/Reference Model (is_ref=True) ---")
    sft_model = VisionModelForCLS(
        device=device, model_name=model_name, num_labels=num_classes,
        ref_cls_weights_path=None, 
        privileged_indices=PRIVILEGED_INDICES_SET, non_privileged_indices=NON_PRIVILEGED_INDICES_SET,
        loss_type=None, is_ref=True, quant_config=quant_config
    ).to(device)
    sft_model.calc_num_params()

    batch_size = 4
    dummy_pixels = torch.rand(batch_size, 3, 224, 224).to(device) 
    dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).to(device).float()
    dummy_sampled_r_indices = torch.randint(0, num_classes, (batch_size,)).to(device)

    print("\n--- Running Forward Pass for FairPO Model ---")
    fairpo_model.train()
    output_dict_fairpo = fairpo_model(dummy_pixels, dummy_labels, sampled_r_indices_for_batch=dummy_sampled_r_indices)

    print("\n--- Running Forward Pass for SFT Model ---")
    sft_model.train()
    output_dict_sft = sft_model(dummy_pixels, dummy_labels) 

    def print_output_dict(name, output_dict):
        print(f"\n--- {name} Results ---")
        print("  Outputs (Probabilities Shape):", output_dict["outputs"].shape)
        print("  Loss Components:")
        for k, v_loss in output_dict["loss"].items():
            print(f"    {k}: {v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss}")
        print("  Accuracy Components:")
        for k_acc_group, v_acc_dict in output_dict["acc"].items():
            if isinstance(v_acc_dict, dict): # Should not happen with current to_loggable
                 print(f"    {k_acc_group}: {v_acc_dict}") # Defensive
            else:
                 print(f"    {k_acc_group}: {v_acc_dict}")

        print("  MAP Components:", output_dict["map"])
        print("  F1 Components:", output_dict["f1"]) # F1 is still placeholder NaNs
        print("  EM Components:", output_dict["em"])

    print_output_dict("FairPO Model", output_dict_fairpo)
    print_output_dict("SFT Model", output_dict_sft)
        
    if os.path.exists(dummy_sft_checkpoint_path):
        os.remove(dummy_sft_checkpoint_path)
        print(f"Removed dummy SFT checkpoint: {dummy_sft_checkpoint_path}")
    print("\nDONE")

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {dev}...")
    main_cls("google/vit-base-patch16-224", device=dev)