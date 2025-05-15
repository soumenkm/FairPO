import os
import sys
from pathlib import Path
from typing import Union, List, Set, Optional, Dict, Any # Added Optional, Dict, Any
import copy
import logging # Added for logging

import torchinfo
import torch
import torch.nn as nn
import torch.nn.functional as F # Import functional API
from transformers import (AutoModel, # Changed from AutoModelForImageClassification
                          BitsAndBytesConfig)
# Assuming Metrics class is in metrics.py and handles detailed metric calculations
from metrics import Metrics # Keep this if you still use it for std_metrics

seed = 42
torch.manual_seed(seed)
import numpy as np # For np.random.choice and np.nan
np.random.seed(seed)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class VisionModelForCLS(torch.nn.Module):
    def __init__(self,
                 device: torch.device,
                 model_name: str,
                 num_labels: int,
                 ref_cls_weights_path: Optional[str], # Can be None
                 privileged_indices: Set[int], # Changed to Set
                 non_privileged_indices: Set[int], # Changed to Set
                 is_ref: bool,
                 loss_type: Optional[str], # Can be None if is_ref is True
                 beta: float = 1.0,
                 epsilon: float = 0.1,
                 quant_config: Union[BitsAndBytesConfig, None] = None):
        super(VisionModelForCLS, self).__init__()
        self.device = device
        self.model_name = model_name
        self.quant_config = quant_config
        self.num_labels = num_labels
        self.ref_cls_wt_path = ref_cls_weights_path
        
        # Ensure these are sets for quick lookups
        self.privileged_indices_set = privileged_indices
        self.non_privileged_indices_set = non_privileged_indices
        
        self.is_ref = is_ref
        self.beta = beta
        self.epsilon = epsilon
        self.eps_log = 1e-8 # Small value for numerical stability in log
        self.loss_type = loss_type # 'dpo', 'simpo', 'cpo' or None

        # Create a boolean mask for quick checking if a label index is privileged
        self.label_is_privileged_mask = torch.zeros(self.num_labels, dtype=torch.bool, device=self.device)
        if self.privileged_indices_set: # Check if not empty
            self.label_is_privileged_mask[list(self.privileged_indices_set)] = True

        # Load the base Vision Transformer model (backbone)
        self.vit = AutoModel.from_pretrained(
            self.model_name,
            quantization_config=self.quant_config,
            # device_map="auto", # device_map might not be ideal if moving parts later
        ).to(self.device) # Move backbone to device
        
        self.d = self.vit.config.hidden_size
        self.c = num_labels
        self.config = self.vit.config # From the base model
        
        # Trainable classifier head (parameters w_t)
        # This will be a single nn.Linear layer now, as per standard practice.
        # The original code had a ModuleList of MLPs, which is highly unconventional
        # for a simple classifier head after a ViT and makes per-label parameters w_t complex.
        # Reverting to a standard single nn.Linear.
        self.classifier_head = nn.Linear(self.d, self.c).to(self.device)
        
        self.ref_classifier_head = None
        if not self.is_ref:
            self.ref_classifier_head = nn.Linear(self.d, self.c).to(self.device)
            if self.ref_cls_wt_path is not None:
                try:
                    # Load the whole state_dict of the reference model
                    ref_checkpoint = torch.load(self.ref_cls_wt_path, map_location=self.device)
                    
                    # Assuming the reference model saved its head as 'classifier_head.weight' and 'classifier_head.bias'
                    # And these are directly under 'model_state_dict'
                    ref_head_state_dict = {}
                    # Need to match the structure of how the SFT model saved its head.
                    # If SFT model was an instance of this class itself (with is_ref=True),
                    # then its classifier head is self.classifier_head.
                    # The keys in checkpoint['model_state_dict'] would be like 'classifier_head.weight'.
                    for k, v in ref_checkpoint["model_state_dict"].items():
                        if k.startswith("classifier_head."): # Match keys for the SFT head
                             ref_head_state_dict[k.replace("classifier_head.", "")] = v
                    
                    if not ref_head_state_dict:
                        logging.warning(f"No keys matching 'classifier_head.*' found in reference checkpoint at {self.ref_cls_wt_path}. Attempting to load from 'model.classifier.*' if it was a HF model.")
                        # Fallback for HF AutoModelForImageClassification structure if that's what was saved
                        for k, v in ref_checkpoint["model_state_dict"].items():
                            if k.startswith("model.classifier."): # HF model usually saves under 'model.classifier'
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
            
            # Freeze the reference classifier head
            if self.ref_classifier_head:
                for param in self.ref_classifier_head.parameters():
                    param.requires_grad = False
                self.ref_classifier_head.eval()

        # Freeze the backbone (ViT) by default
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze last layer of ViT if not reference training (optional, for some fine-tuning)
        # if not self.is_ref:
        #     for param in self.vit.encoder.layer[-1].parameters():
        #         param.requires_grad = True

        # Ensure trainable classifier head parameters require gradients
        for param in self.classifier_head.parameters():
            param.requires_grad = True


    def calc_num_params(self) -> None:
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(self) # Prints the model structure
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
        
        # 1. Get hidden state from backbone
        # Ensure backbone computation doesn't track gradients if it's fully frozen
        # or only parts are unfrozen.
        # The `param.requires_grad = False` loop in __init__ handles the vit part.
        # If some vit layers are unfrozen, their gradients will be computed.
        vit_outputs = self.vit(pixel_values=pixels.to(self.device))
        # Use last_hidden_state and then usually the [CLS] token embedding (index 0)
        embeddings = vit_outputs.last_hidden_state[:, 0, :] # Shape (b, d)
        embeddings = embeddings.to(torch.float32)

        # 2. Get logits from the *trainable* classifier head
        logits_policy = self.classifier_head(embeddings) # (b, c)
        
        logits_ref = None
        if self.ref_classifier_head:
            with torch.no_grad():
                logits_ref = self.ref_classifier_head(embeddings) # (b, c)

        # --- Initialize Loss Dictionary ---
        # These will store L_P^(s) and L_NP^(s) - the average loss for priv/non-priv components for the batch
        loss_dict_components = {'privileged': torch.tensor(0.0, device=self.device), 
                                'non_privileged': torch.tensor(0.0, device=self.device)}
        
        if self.is_ref: # SFT Training (Standard BCE over all labels)
            # For SFT, labels are (b, c). logits_policy are (b,c)
            sft_loss = nn.BCEWithLogitsLoss()(logits_policy, labels.float().to(self.device))
            loss_dict_components['loss'] = sft_loss # For SFT, 'loss' is the main loss
        
        else: # FairPO Training (new logic based on sampled_r_indices_for_batch)
            if sampled_r_indices_for_batch is None:
                logging.error("sampled_r_indices_for_batch is None during FairPO training! This should not happen.")
                # Fallback: return zero losses or raise error, metrics will still be computed.
                # Returning zero losses can mask issues, but allows training to proceed if this is intermittent.
                pass # loss_dict_components already initialized to 0.0
            else:
                actual_privileged_losses_this_batch = []
                actual_non_privileged_losses_this_batch = []
                
                current_labels_device = labels.to(self.device)

                for i in range(batch_size): # For each sample in the batch
                    r_idx = sampled_r_indices_for_batch[i].item() # The label sampled for THIS instance
                    is_r_privileged = self.label_is_privileged_mask[r_idx].item()

                    # Get relevant logits and labels for the current sample i
                    current_sample_labels = current_labels_device[i]       # Shape (c,)
                    current_sample_logits_policy = logits_policy[i] # Shape (c,)
                    current_sample_logits_ref = logits_ref[i] if logits_ref is not None else None # Shape (c,)

                    if is_r_privileged:
                        l_idx = r_idx # This is our privileged label 'l'
                        gt_label_l = current_sample_labels[l_idx]       # Scalar (0 or 1)
                        policy_logit_l = current_sample_logits_policy[l_idx] # Scalar
                        
                        ref_logit_l = torch.tensor(0.0, device=self.device) # Default if no ref
                        if current_sample_logits_ref is not None:
                            ref_logit_l = current_sample_logits_ref[l_idx]

                        loss_for_this_priv_sample = torch.tensor(0.0, device=self.device)
                        confusing_set_found_and_processed = False

                        # Probabilities for the privileged label l
                        prob_policy_l = torch.sigmoid(policy_logit_l)
                        prob_ref_l = torch.sigmoid(ref_logit_l)

                        if gt_label_l == 1: # Privileged label l_idx is TRUE POSITIVE
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
                                
                                # DPO term: h(l, k) for l > k
                                log_term_policy_l_vs_ref = torch.log(prob_policy_l + self.eps_log) - torch.log(prob_ref_l + self.eps_log)
                                log_term_policy_k_vs_ref = torch.log(prob_policy_k + self.eps_log) - torch.log(prob_ref_k + self.eps_log)
                                h_val = log_term_policy_l_vs_ref - log_term_policy_k_vs_ref
                                loss_for_this_priv_sample = -torch.log(torch.sigmoid(self.beta * h_val) + self.eps_log)

                        elif gt_label_l == 0: # Privileged label l_idx is TRUE NEGATIVE
                            confusing_positives_k_prime_indices = []
                            for k_prime_candidate_idx in range(self.num_labels):
                                if k_prime_candidate_idx == l_idx: continue

                                gt_label_k_prime_candidate = current_sample_labels[k_prime_candidate_idx]
                                policy_logit_k_prime_candidate = current_sample_logits_policy[k_prime_candidate_idx]
                                prob_policy_k_prime_candidate = torch.sigmoid(policy_logit_k_prime_candidate)

                                if gt_label_k_prime_candidate == 1 and prob_policy_k_prime_candidate <= prob_policy_l: # Note: score of positive k' is LEQ score of negative l
                                    confusing_positives_k_prime_indices.append(k_prime_candidate_idx)
                            
                            if confusing_positives_k_prime_indices:
                                confusing_set_found_and_processed = True
                                sampled_k_prime_idx = np.random.choice(confusing_positives_k_prime_indices)

                                policy_logit_k_prime = current_sample_logits_policy[sampled_k_prime_idx]
                                ref_logit_k_prime = current_sample_logits_ref[sampled_k_prime_idx] if current_sample_logits_ref is not None else torch.tensor(0.0, device=self.device)
                                
                                prob_policy_k_prime = torch.sigmoid(policy_logit_k_prime)
                                prob_ref_k_prime = torch.sigmoid(ref_logit_k_prime)

                                # DPO term: h(k', l) for k' > l (true neg l is dispreferred to confusing pos k')
                                log_term_policy_k_prime_vs_ref = torch.log(prob_policy_k_prime + self.eps_log) - torch.log(prob_ref_k_prime + self.eps_log)
                                log_term_policy_l_vs_ref = torch.log(prob_policy_l + self.eps_log) - torch.log(prob_ref_l + self.eps_log)
                                h_val = log_term_policy_k_prime_vs_ref - log_term_policy_l_vs_ref
                                loss_for_this_priv_sample = -torch.log(torch.sigmoid(self.beta * h_val) + self.eps_log)
                        
                        if not confusing_set_found_and_processed: # Fallback to BCE for this privileged label l_idx
                            bce_loss_fn_single = torch.nn.BCEWithLogitsLoss() # reduction='mean' is default, on single logit/label
                            loss_for_this_priv_sample = bce_loss_fn_single(policy_logit_l.unsqueeze(0), gt_label_l.float().unsqueeze(0))
                        
                        actual_privileged_losses_this_batch.append(loss_for_this_priv_sample)

                    else: # r_idx is a non-privileged label
                        j_idx = r_idx
                        gt_label_j = current_sample_labels[j_idx]
                        policy_logit_j = current_sample_logits_policy[j_idx]
                        ref_logit_j = current_sample_logits_ref[j_idx] if current_sample_logits_ref is not None else torch.tensor(0.0, device=self.device)

                        bce_loss_fn_single = torch.nn.BCEWithLogitsLoss()
                        loss_policy_j = bce_loss_fn_single(policy_logit_j.unsqueeze(0), gt_label_j.float().unsqueeze(0))
                        loss_ref_j = bce_loss_fn_single(ref_logit_j.unsqueeze(0), gt_label_j.float().unsqueeze(0)) # Detached by no_grad on ref_classifier
                        
                        loss_val_non_priv = torch.relu(loss_policy_j - loss_ref_j.detach() - self.epsilon) # Ensure ref_loss is detached
                        actual_non_privileged_losses_this_batch.append(loss_val_non_priv)

                # Store the mean of actual losses computed (L_P^(s) and L_NP^(s))
                if actual_privileged_losses_this_batch: # If any privileged samples were processed
                    loss_dict_components['privileged'] = torch.mean(torch.stack(actual_privileged_losses_this_batch))
                # else: loss_dict_components['privileged'] remains 0.0
                
                if actual_non_privileged_losses_this_batch: # If any non-privileged samples were processed
                    loss_dict_components['non_privileged'] = torch.mean(torch.stack(actual_non_privileged_losses_this_batch))
                # else: loss_dict_components['non_privileged'] remains 0.0
        
        # --- Calculate Standard Metrics (on all labels, for evaluation) ---
        # These metrics are independent of the sampled 'r' logic for loss computation.
        # They use the full logits_policy and labels.
        # You'll need a separate Metrics class or functions for these.
        metrics_output = self.calculate_std_metrics(logits_policy, labels.to(self.device))

        # The 'loss' key in the returned dict should contain the components for GRPO or the SFT loss
        return {
            "outputs": torch.sigmoid(logits_policy), # Return probabilities
            "logits": logits_policy, # Also return logits for potential use
            "loss": loss_dict_components,  # Dict with 'privileged', 'non_privileged' or 'loss' (for SFT)
            "acc": metrics_output['acc'],
            "f1": metrics_output['f1'],
            "map": metrics_output['map'],
            "em": metrics_output['em']
        }

    def calculate_std_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """
        Calculates standard multi-label classification metrics.
        logits: (batch_size, num_labels) - raw logits from the policy model
        labels: (batch_size, num_labels) - ground truth labels
        """
        # Using your Metrics class structure if it's available and does this
        # Otherwise, implementing basic versions here or calling your Metrics methods.
        # For simplicity, let's assume Metrics class has methods that take (probs, labels, priv_indices, non_priv_indices)
        
        # Ensure labels are on the same device as logits
        labels = labels.to(logits.device)
        prob_scores = torch.sigmoid(logits) # Calculate probabilities once

        # If your Metrics class is designed like in the original `models.py`
        # accuracy_components = Metrics.compute_accuracy_components(
        #     prob_scores=prob_scores, labels=labels,
        #     privileged_indices=list(self.privileged_indices_set), # Metrics might expect list
        #     non_privileged_indices=list(self.non_privileged_indices_set)
        # )
        # map_components = Metrics.compute_map_components(prob_scores, labels, list(self.privileged_indices_set), list(self.non_privileged_indices_set))
        # f1_components = Metrics.compute_f1_components(prob_scores, labels, list(self.privileged_indices_set), list(self.non_privileged_indices_set))
        # em_components = Metrics.compute_exact_match_accuracy_components(prob_scores, labels, list(self.privileged_indices_set), list(self.non_privileged_indices_set))

        # Placeholder implementation if Metrics class methods are not directly usable or need adaptation:
        # This is a simplified version. You should use a robust metrics library (like torchmetrics or scikit-learn)
        # or ensure your Metrics class is correctly implemented.
        
        preds = (prob_scores > 0.5).float()
        
        overall_acc = (preds == labels).float().mean().item()
        exact_match = (torch.all(preds == labels, dim=1)).float().mean().item()

        priv_indices_list = list(self.privileged_indices_set) if self.privileged_indices_set else []
        non_priv_indices_list = list(self.non_privileged_indices_set) if self.non_privileged_indices_set else []

        acc_privileged = torch.tensor(np.nan, device=logits.device)
        em_privileged = torch.tensor(np.nan, device=logits.device)
        if priv_indices_list:
            if preds[:, priv_indices_list].numel() > 0 : # Check if slice is not empty
                acc_privileged = (preds[:, priv_indices_list] == labels[:, priv_indices_list]).float().mean()
                em_privileged = (torch.all(preds[:, priv_indices_list] == labels[:, priv_indices_list], dim=1)).float().mean()

        acc_non_privileged = torch.tensor(np.nan, device=logits.device)
        em_non_privileged = torch.tensor(np.nan, device=logits.device)
        if non_priv_indices_list:
            if preds[:, non_priv_indices_list].numel() > 0:
                acc_non_privileged = (preds[:, non_priv_indices_list] == labels[:, non_priv_indices_list]).float().mean()
                em_non_privileged = (torch.all(preds[:, non_priv_indices_list] == labels[:, non_priv_indices_list], dim=1)).float().mean()
        
        # F1 and mAP are more complex and typically require library functions for reliable calculation
        # For now, returning NaNs or simple proxies.
        # You SHOULD replace these with proper calculations using torchmetrics or sklearn.metrics.f1_score, average_precision_score
        f1_overall = torch.tensor(np.nan, device=logits.device) # Placeholder
        map_overall = torch.tensor(np.nan, device=logits.device) # Placeholder
        f1_privileged = torch.tensor(np.nan, device=logits.device)
        map_privileged = torch.tensor(np.nan, device=logits.device)
        f1_non_privileged = torch.tensor(np.nan, device=logits.device)
        map_non_privileged = torch.tensor(np.nan, device=logits.device)


        # Convert scalar tensors to Python floats/np.nan for logging
        def to_loggable(val):
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else val.tolist() # Or handle multi-element tensors appropriately
            return val

        return {
            'acc': {'acc': to_loggable(overall_acc), 
                    'privileged': to_loggable(acc_privileged), 
                    'non_privileged': to_loggable(acc_non_privileged)},
            'em': {'em': to_loggable(exact_match), 
                   'em_privileged': to_loggable(em_privileged), 
                   'em_non_privileged': to_loggable(em_non_privileged)},
            'f1': {'f1': to_loggable(f1_overall), # Note: original code had 'macro_f1', adjust if needed
                   'f1_privileged': to_loggable(f1_privileged), 
                   'f1_non_privileged': to_loggable(f1_non_privileged)},
            'map': {'mAP': to_loggable(map_overall), 
                    'mAP_privileged': to_loggable(map_privileged), 
                    'mAP_non_privileged': to_loggable(map_non_privileged)}
        }
        
def main_cls(model_name: str, device: torch.device) -> None:
    # Example usage
    logging.basicConfig(level=logging.INFO) # Setup logging for the example
    quant_config = None

    num_classes = 80
    ALL_LABELS_INDICES = list(range(80))
    # Define privileged as first 20 for testing, non-privileged as the rest
    PRIVILEGED_INDICES_SET = set(ALL_LABELS_INDICES[:20]) 
    NON_PRIVILEGED_INDICES_SET = set(ALL_LABELS_INDICES) - PRIVILEGED_INDICES_SET

    print(f"Privileged Indices: {sorted(list(PRIVILEGED_INDICES_SET))}")
    print(f"Non-Privileged Indices: {sorted(list(NON_PRIVILEGED_INDICES_SET))}")

    # --- Test FairPO model (is_ref=False) ---
    print("\n--- Initializing FairPO Model (is_ref=False) ---")
    # Create a dummy SFT weight file for the FairPO model to load as reference
    # This SFT model would typically be trained first and its classifier head saved.
    sft_model_for_ref = VisionModelForCLS(
        device=device, model_name=model_name, num_labels=num_classes,
        ref_cls_weights_path=None, privileged_indices=set(), non_privileged_indices=set(),
        is_ref=True, loss_type=None 
    ).to(device)
    dummy_sft_checkpoint_path = "dummy_sft_ref_checkpoint.pth"
    torch.save({'model_state_dict': sft_model_for_ref.state_dict()}, dummy_sft_checkpoint_path)
    print(f"Saved dummy SFT checkpoint for reference: {dummy_sft_checkpoint_path}")


    fairpo_model = VisionModelForCLS(
        device=device,
        model_name=model_name,
        num_labels=num_classes,
        ref_cls_weights_path=dummy_sft_checkpoint_path, 
        privileged_indices=PRIVILEGED_INDICES_SET,
        non_privileged_indices=NON_PRIVILEGED_INDICES_SET,
        loss_type="dpo", 
        beta=2.0,    
        epsilon=0.01, 
        is_ref=False, 
        quant_config=quant_config
    ).to(device)
    fairpo_model.calc_num_params()

    # --- Test SFT/Reference model (is_ref=True) ---
    print("\n--- Initializing SFT/Reference Model (is_ref=True) ---")
    sft_model = VisionModelForCLS(
        device=device,
        model_name=model_name,
        num_labels=num_classes,
        ref_cls_weights_path=None, 
        privileged_indices=PRIVILEGED_INDICES_SET, # These are not used by SFT loss but ok to pass
        non_privileged_indices=NON_PRIVILEGED_INDICES_SET,
        loss_type=None, # Not used
        is_ref=True, 
        quant_config=quant_config
    ).to(device)
    sft_model.calc_num_params()


    # Dummy input data
    batch_size = 4
    dummy_pixels = torch.rand(batch_size, 3, 224, 224).to(device) 
    dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).to(device).float() # Ensure float for BCE
    
    # For FairPO, we need sampled_r_indices
    dummy_sampled_r_indices = torch.randint(0, num_classes, (batch_size,)).to(device)

    print("\n--- Running Forward Pass for FairPO Model ---")
    fairpo_model.train() # Set to train mode to compute losses
    output_dict_fairpo = fairpo_model(dummy_pixels, dummy_labels, sampled_r_indices_for_batch=dummy_sampled_r_indices)

    print("\n--- Running Forward Pass for SFT Model ---")
    sft_model.train()
    # SFT model forward doesn't take sampled_r_indices for its loss logic
    output_dict_sft = sft_model(dummy_pixels, dummy_labels) 


    def print_output_dict(name, output_dict):
        print(f"\n--- {name} Results ---")
        print("  Outputs (Probabilities Shape):", output_dict["outputs"].shape)
        print("  Loss Components:")
        for k, v in output_dict["loss"].items():
            print(f"    {k}: {v.item() if isinstance(v, torch.Tensor) else v}")
        print("  Accuracy Components:")
        for k, v_dict in output_dict["acc"].items():
             print(f"    {k}: {v_dict}")
        # Similarly for map, f1, em if they are dicts of dicts
        print("  MAP Components:", output_dict["map"])
        print("  F1 Components:", output_dict["f1"])
        print("  EM Components:", output_dict["em"])

    print_output_dict("FairPO Model", output_dict_fairpo)
    print_output_dict("SFT Model", output_dict_sft)
        
    # Clean up dummy checkpoint
    if os.path.exists(dummy_sft_checkpoint_path):
        os.remove(dummy_sft_checkpoint_path)
        print(f"Removed dummy SFT checkpoint: {dummy_sft_checkpoint_path}")

    print("\nDONE")

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {dev}...")
    main_cls("google/vit-base-patch16-224", device=dev)