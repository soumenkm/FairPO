import os
import sys
from pathlib import Path
from typing import Union, List, Set
import copy

import torchinfo
import torch
import torch.nn as nn
import torch.nn.functional as F # Import functional API
from transformers import (AutoModelForImageClassification,
                          BitsAndBytesConfig)

seed = 42
torch.manual_seed(seed)

if __name__ == "__main__": # Commented out for execution in interactive env
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class VisionModelForCLS(torch.nn.Module):
    def __init__(self,
                 device: torch.device,
                 model_name: str,
                 num_labels: int,
                 ref_cls_weights_path: str, # New: Path to SFT classifier weights
                 privileged_indices: List[int], # New: Indices for P
                 non_privileged_indices: List[int], # New: Indices for P_bar
                 is_ref: bool, # New: Flag for reference model
                 beta: float = 1.0,       # New: DPO beta hyperparameter
                 epsilon: float = 0.1,    # New: Constraint slack epsilon
                 quant_config: Union[BitsAndBytesConfig, None] = None): # Added default
        super(VisionModelForCLS, self).__init__()
        self.device = device
        self.model_name = model_name
        self.quant_config = quant_config
        self.num_labels = num_labels
        self.ref_cls_wt_path = ref_cls_weights_path
        self.privileged_indices = privileged_indices
        self.non_privileged_indices = non_privileged_indices
        self.is_ref = is_ref
        self.beta = beta
        self.epsilon = epsilon
        self.eps = 1e-8 # Small value for numerical stability

        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            quantization_config=self.quant_config,
            device_map="auto",
        )
        self.d = self.model.config.hidden_size
        self.c = num_labels
        self.config = self.model.config
        
        # Trainable classifier head (parameters w_t)
        self.model.classifier = self._get_classifier_head(self.d, self.c).to(self.device)
            
        if not self.is_ref:
            # Reference Model Classifier Head (parameters hat{w}_t)
            self.model.ref_classifier = self._get_classifier_head(self.d, self.c).to(self.device)
            if self.ref_cls_wt_path is not None:
                ref_cls_state_dict = torch.load(self.ref_cls_wt_path, map_location=self.device)
                ref_cls_state_dict_new = copy.deepcopy(ref_cls_state_dict)
                ref_cls_state_dict_new["model_state"] = {}
                for k, v in ref_cls_state_dict["model_state"].items():
                    # Replace "classifier." with "ref_classifier." in the state dict keys
                    if "classifier." in k:
                        k = k.replace("classifier.", "ref_classifier.")
                    ref_cls_state_dict_new["model_state"][k] = v
                self.load_state_dict(ref_cls_state_dict_new["model_state"], strict=False) # !TODO: Load ref model weights
            else:
                print("WARNING: Reference classifier weights are NOT loaded. Using initialized weights.")
            # Freeze the reference classifier head
            for param in self.model.ref_classifier.parameters():
                param.requires_grad = False

        # Freeze the backbone (ViT)
        for param in self.model.vit.parameters():
            param.requires_grad = False
        
        # Ensure trainable classifier head parameters require gradients
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def _get_classifier_head(self, d: int, num_labels: int) -> torch.nn.Module:
        # Creates a ModuleList containing one MLP per label
        mlp_list = []
        for i in range(num_labels):
            # Using Sequential for each MLP head
            mlp = torch.nn.Sequential(
                torch.nn.Linear(d, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1)
            )
            mlp_list.append(mlp)
        cls_head = torch.nn.ModuleList(mlp_list)
        return cls_head

    def _get_scores(self, classifier_head: nn.ModuleList, hidden_state: torch.tensor) -> torch.tensor:
        """Applies the classifier head to get logits, then applies sigmoid.
        hidden_state.shape = (b, d)
        """
        batch_size = hidden_state.shape[0]
        logits = torch.zeros((batch_size, self.c)).to(self.device)
        for i, layer in enumerate(classifier_head):
            # layer output shape is (batch_size, 1)
            logits[:, i] = layer(hidden_state).squeeze(-1) # Squeeze the last dimension
        # Apply sigmoid to get probabilities m(x; w)
        prob_scores = torch.sigmoid(logits)
        return prob_scores # (b, c)

    def calc_num_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        
        torchinfo.summary(self.model) # Can be verbose with ModuleList
        print(f"Backbone: {self.model_name}")
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters (classifier heads): {train_params}")
        if total_params > 0:
           print(f"Training Percentage: {train_params * 100 / total_params:.3f}%")
        else:
           print("Training Percentage: N/A (total_params is zero)")

    def _compute_privileged_loss(self,
                                 prob_scores: torch.tensor,
                                 ref_prob_scores: torch.tensor,
                                 labels: torch.tensor) -> torch.tensor:
        """Computes L_privileged averaged over the batch.
        prob_scores.shape = (b, c), ref_prob_scores.shape = (b, c), labels.shape = (b, c)"""
        
        batch_size = prob_scores.shape[0]
        total_priv_loss = 0.0
        num_pairs = 0

        for i in range(batch_size):
            current_scores_i = prob_scores[i] # Scores for instance i (shape [c])
            ref_scores_i = ref_prob_scores[i] # Ref scores for instance i (shape [c])
            labels_i = labels[i]              # Labels for instance i (shape [c])

            # Find positive privileged labels for this instance
            positive_priv_labels = [
                l for l in self.privileged_indices if labels_i[l] == 1
            ]

            for l in positive_priv_labels:
                # Find confusing negatives k for this positive privileged label l
                # k is confusing if y_ik = 0 and m(x_i; w_k) >= m(x_i; w_l)
                confusing_negatives = [
                    k for k in range(self.c)
                    if labels_i[k] == 0 and current_scores_i[k] >= current_scores_i[l]
                ]

                for k in confusing_negatives:
                    # Get scores for l and k
                    m_il = current_scores_i[l]
                    m_ik = current_scores_i[k]
                    ref_m_il = ref_scores_i[l]
                    ref_m_ik = ref_scores_i[k]

                    # Calculate h_w(x_i, l, k)
                    log_term_l = torch.log(m_il / (ref_m_il + self.eps) + self.eps)
                    log_term_k = torch.log(m_ik / (ref_m_ik + self.eps) + self.eps)
                    h_w = log_term_l - log_term_k

                    # Calculate loss term: -log(sigmoid(beta * h_w))
                    loss_term = -F.logsigmoid(self.beta * h_w) # More stable than log(sigmoid())

                    total_priv_loss += loss_term
                    num_pairs += 1

        # Average loss over all (l, k) pairs found in the batch
        if num_pairs == 0:
            # Return zero loss tensor if no privileged pairs found
            # Ensure it's on the correct device and requires grad if necessary
            # (though loss itself shouldn't require grad, the computation path should)
            return torch.tensor(0.0, device=self.device, requires_grad=prob_scores.requires_grad)
        else:
            return total_priv_loss / num_pairs

    def _compute_non_privileged_loss(self,
                                     prob_scores: torch.tensor,
                                     ref_prob_scores: torch.tensor,
                                     labels: torch.tensor) -> torch.tensor:
        """Computes L_nonprivileged averaged over the batch.
        prob_scores.shape = (b, c), ref_prob_scores.shape = (b, c), labels.shape = (b, c)"""
        
        # Select scores and labels only for non-privileged indices
        non_priv_indices_tensor = torch.tensor(self.non_privileged_indices, device=self.device, dtype=torch.long)

        m_nonpriv = prob_scores[:, non_priv_indices_tensor] # Shape (b, |P_bar|)
        ref_m_nonpriv = ref_prob_scores[:, non_priv_indices_tensor] # Shape (b, |P_bar|)
        y_nonpriv = labels[:, non_priv_indices_tensor] # Shape (b, |P_bar|)

        # Calculate BCE loss for current model (element-wise)
        # Use clamp to prevent log(0) with probabilities from sigmoid
        loss_current = F.binary_cross_entropy(
            m_nonpriv.clamp(min=self.eps, max=1.0-self.eps),
            y_nonpriv.to(torch.float32),
            reduction='none' # Get loss per element
        ) # (b, |P_bar|)

        # Calculate BCE loss for reference model (element-wise)
        # No gradient needed for reference calculation part
        with torch.no_grad():
             loss_ref = F.binary_cross_entropy(
                 ref_m_nonpriv.clamp(min=self.eps, max=1.0-self.eps),
                 y_nonpriv.to(torch.float32),
                 reduction='none' # Get loss per element
             ) # (b, |P_bar|)

        # Calculate hinge loss element-wise
        # max(0, loss(w_j) - loss(hat{w}_j) - epsilon)
        hinge_loss = torch.relu(loss_current - loss_ref - self.epsilon) # (b, |P_bar|)

        # Average the hinge loss over all batch elements and non-privileged labels
        avg_non_priv_loss = torch.mean(hinge_loss)  # (scalar)
        return avg_non_priv_loss

    def _compute_ref_model_loss(self,
                                prob_scores: torch.tensor,
                                labels: torch.tensor) -> torch.tensor:
        """Computes reference model loss averaged over the batch.
        prob_scores.shape = (b, c), labels.shape = (b, c)"""
        
        # Calculate BCE loss for current model (element-wise)
        # Use clamp to prevent log(0) with probabilities from sigmoid
        loss_current = F.binary_cross_entropy(
            prob_scores.clamp(min=self.eps, max=1.0-self.eps),
            labels.to(torch.float32),
            reduction='none' # Get loss per element
        ) # (b, c) 

        # Average the loss over all batch elements and labels
        avg_ref_model_loss = torch.mean(loss_current)  # (scalar)
        return avg_ref_model_loss

    def _compute_loss(self,
                      prob_scores: torch.tensor,
                      ref_prob_scores: torch.tensor,
                      labels: torch.tensor) -> dict:
        """
        Computes both privileged and non-privileged loss components.
        prob_scores.shape = (b, c), ref_prob_scores.shape = (b, c), labels.shape = (b, c)
        """
        if not self.is_ref:
            loss_privileged = self._compute_privileged_loss(prob_scores, ref_prob_scores, labels)
            loss_non_privileged = self._compute_non_privileged_loss(prob_scores, ref_prob_scores, labels)
            return {
                "privileged": loss_privileged,
                "non_privileged": loss_non_privileged
            }
        else:
            loss_ref_model = self._compute_ref_model_loss(prob_scores, labels)
            return {
                "loss": loss_ref_model
            }
    
    def _compute_accuracy_subset(self,
                                 prob_scores: torch.Tensor,
                                 labels: torch.Tensor,
                                 indices: List[int],
                                 threshold: float = 0.5) -> float:
        """Computes element-wise accuracy for a given subset of label indices."""
        if len(indices) == 0:
            return float('nan') # Not applicable if no labels in the subset

        # Ensure labels are on the same device and float type
        labels = labels.to(device=self.device, dtype=torch.float32)

        # Select relevant columns for the subset
        scores_subset = prob_scores[:, indices] # (b, |subset|)
        labels_subset = labels[:, indices]      # (b, |subset|)

        # Get binary predictions
        predictions_subset = (scores_subset >= threshold).float() # (b, |subset|)

        # Compare predictions with true labels element-wise
        correct = (predictions_subset == labels_subset) # (b, |subset|) boolean

        # Calculate accuracy (mean over all elements in the subset across the batch)
        accuracy = correct.float().mean().item() # .item() to get Python float
        return accuracy

    def _compute_accuracy_privileged(self,
                                   prob_scores: torch.Tensor,
                                   labels: torch.Tensor,
                                   threshold: float = 0.5) -> float:
        """Computes element-wise accuracy for the privileged labels."""
        return self._compute_accuracy_subset(prob_scores, labels, self.privileged_indices, threshold)

    def _compute_accuracy_non_privileged(self,
                                       prob_scores: torch.Tensor,
                                       labels: torch.Tensor,
                                       threshold: float = 0.5) -> float:
        """Computes element-wise accuracy for the non-privileged labels."""
        return self._compute_accuracy_subset(prob_scores, labels, self.non_privileged_indices, threshold)

    def _compute_accuracy_overall(self,
                                prob_scores: torch.Tensor,
                                labels: torch.Tensor,
                                threshold: float = 0.5) -> float:
        """Computes element-wise accuracy for ALL labels."""
        # Ensure labels are on the same device and float type
        labels = labels.to(device=self.device, dtype=torch.float32)
        predictions = (prob_scores >= threshold).float()
        correct = (predictions == labels)
        accuracy = correct.float().mean().item()
        return accuracy
        
    def forward(self, pixels: torch.tensor, labels: Union[torch.tensor, None]) -> dict:
        """
        pixels.shape = (b, 3, H, W) e.g., (b, 3, 224, 224)
        labels.shape = (b, c) if provided, else None
        """
        
        # 1. Get hidden state from backbone
        # Ensure pixels is on the right device if not already
        pixels = pixels.to(self.device) 

        # Get hidden states from the frozen backbone
        with torch.no_grad(): # Ensure backbone computation doesn't track gradients
            outputs = self.model.vit(pixel_values=pixels, output_hidden_states=False) # Don't need all hidden states
            hidden_state = outputs.last_hidden_state[:,0,:] # Shape (b, d)
            
        # Ensure hidden_state is float32 for classifier heads if they expect it
        hidden_state = hidden_state.to(torch.float32) # (b, d)

        # 2. Get probability scores from the *trainable* classifier head
        prob_scores = self._get_scores(self.model.classifier, hidden_state) # (b, c)

        if not self.is_ref:
            # 3. Get probability scores from the *reference* classifier head
            with torch.no_grad(): # No gradients needed for reference model
                ref_prob_scores = self._get_scores(self.model.ref_classifier, hidden_state) # (b, c)

        # 4. Compute loss components if labels are provided
        loss_components = None
        accuracy_threshold = 0.5 # Default threshold for binary classification
        if labels is not None:
            labels = labels.to(self.device) # Ensure labels are on the correct device
            if not self.is_ref:
                loss_components = self._compute_loss(prob_scores, ref_prob_scores, labels)
                accuracy_components = {
                    "privileged": self._compute_accuracy_privileged(prob_scores, labels, accuracy_threshold),
                    "non_privileged": self._compute_accuracy_non_privileged(prob_scores, labels, accuracy_threshold),
                    "acc": self._compute_accuracy_overall(prob_scores, labels, accuracy_threshold)
                }
            else:
                loss_components = self._compute_loss(prob_scores, None, labels)
                accuracy_components = {
                    "acc": self._compute_accuracy_overall(prob_scores, labels, accuracy_threshold)
                }
            
        if self.training and loss_components is None:
            raise ValueError("Labels must be provided during training to compute loss components.")

        # 5. Return results
        # The outer training loop will use loss_components with alpha weights
        return {"outputs": prob_scores, "loss": loss_components, "acc": accuracy_components}

def main_cls(model_name: str, device: torch.device) -> None:
    # Example usage
    quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
    ) if torch.cuda.is_available() else None # Only use quantization if CUDA is available

    num_classes = 14
    # Assume these are defined based on your specific task
    # Example: For NIH Chest X-ray14 with 14 labels
    ALL_LABELS = list(range(14))
    PRIVILEGED_INDICES_SET = {0, 5, 10} # Example: Indices of 'Mass', 'Pneumothorax', etc.
    NON_PRIVILEGED_INDICES_SET = set(ALL_LABELS) - PRIVILEGED_INDICES_SET

    # Convert sets to lists/tensors for easier indexing if needed
    PRIVILEGED_INDICES = sorted(list(PRIVILEGED_INDICES_SET))
    NON_PRIVILEGED_INDICES = sorted(list(NON_PRIVILEGED_INDICES_SET))
    print(f"Privileged Indices: {PRIVILEGED_INDICES}")
    print(f"Non-Privileged Indices: {NON_PRIVILEGED_INDICES}")

    # Initialize the model
    model = VisionModelForCLS(
        device=device,
        model_name=model_name,
        num_labels=num_classes,
        ref_cls_weights_path=None, # "sft_classifier_weights.pth", # Example path
        privileged_indices=PRIVILEGED_INDICES,
        non_privileged_indices=NON_PRIVILEGED_INDICES,
        beta=1.0,      # Example value
        epsilon=0.1,   # Example value
        is_ref=False,  # Set to True for reference model
        quant_config=quant_config
    ).to(device)

    # TODO: --- !!! IMPORTANT !!! ---
    # Load your SFT classifier weights into model.ref_classifier here
    # Example (replace with actual loading):
    # try:
    #     model.ref_classifier.load_state_dict(torch.load('sft_classifier_weights.pth', map_location=device))
    #     print("Loaded reference classifier weights.")
    # except FileNotFoundError:
    #     print("WARNING: sft_classifier_weights.pth not found. Reference model uses initial weights.")
    # --------------------------

    model.calc_num_params()

    # Dummy input data
    batch_size = 4
    dummy_pixels = torch.rand(batch_size, 3, 224, 224).to(device) # Random pixel values
    # Dummy labels (multi-label, binary)
    dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).to(device)

    print("\n--- Running Forward Pass ---")
    # Run forward pass
    output_dict = model(dummy_pixels, dummy_labels)

    print("\n--- Output ---")
    print("Output Scores Shape:", output_dict["outputs"].shape)

    if output_dict["loss"] is not None:
        print("\n--- Loss Components ---")
        print(output_dict["loss"])
        print("\n--- Accuracy Components ---")
        print(output_dict["acc"])
    else:
        print("Not computed (labels were None).")
        
    print("\nDONE")

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {dev}...")
    # Using a smaller model for potentially faster testing if needed
    main_cls("google/vit-base-patch16-224", device=dev) # Or use the base model