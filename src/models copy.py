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
from metrics import Metrics

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
                 loss_type: str, # "dpo", "simpo", "cpo" or None if is_ref is True
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
        self.loss_type = loss_type

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
                ref_cls_state_dict = torch.load(self.ref_cls_wt_path, map_location=self.device, weights_only=False)
                ref_cls_state_dict_new = {}
                for k, v in ref_cls_state_dict["model_state_dict"].items():
                    # Replace "classifier." with "ref_classifier." in the state dict keys
                    if "model.classifier." in k:
                        k = k.replace("model.classifier.", "")
                        ref_cls_state_dict_new[k] = v
                self.model.ref_classifier.load_state_dict(ref_cls_state_dict_new, strict=True) 
                print(f"INFO: Reference classifier weights loaded from {self.ref_cls_wt_path}")
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
            
        if not self.is_ref:
            for param in self.model.vit.encoder.layer[-1].parameters():
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
        
        print(self)
        # torchinfo.summary(self.model) # Can be verbose with ModuleList
        print(f"Backbone: {self.model_name}")
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters (classifier heads): {train_params}")
        if total_params > 0:
           print(f"Training Percentage: {train_params * 100 / total_params:.3f}%")
        else:
           print("Training Percentage: N/A (total_params is zero)")
        
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

        ref_prob_scores = None
        if not self.is_ref:
            # 3. Get probability scores from the *reference* classifier head
            with torch.no_grad(): # No gradients needed for reference model
                ref_prob_scores = self._get_scores(self.model.ref_classifier, hidden_state) # (b, c)

        # 4. === Compute Loss and Accuracy Components Separately ===
        loss_components = Metrics.compute_loss_components(
            prob_scores=prob_scores,
            ref_prob_scores=ref_prob_scores,
            labels=labels,
            privileged_indices=self.privileged_indices,
            non_privileged_indices=self.non_privileged_indices,
            is_ref_mode=self.is_ref,
            beta=self.beta,
            epsilon=self.epsilon,
            loss_type=self.loss_type
        )

        accuracy_components = Metrics.compute_accuracy_components(
            prob_scores=prob_scores,
            labels=labels,
            privileged_indices=self.privileged_indices,
            non_privileged_indices=self.non_privileged_indices
        )
        
        map_components = Metrics.compute_map_components(
            prob_scores=prob_scores,
            labels=labels,
            privileged_indices=self.privileged_indices,
            non_privileged_indices=self.non_privileged_indices
        )
        
        f1_components = Metrics.compute_f1_components(
            prob_scores=prob_scores,
            labels=labels,
            privileged_indices=self.privileged_indices,
            non_privileged_indices=self.non_privileged_indices
        )
        
        em_components = Metrics.compute_exact_match_accuracy_components(
            prob_scores=prob_scores,
            labels=labels,
            privileged_indices=self.privileged_indices,
            non_privileged_indices=self.non_privileged_indices
        )
        
        # 5. Check label requirement during training
        if self.training and labels is None:
            raise ValueError("Labels must be provided during training.")

        # 6. Return results
        return {
            "outputs": prob_scores,
            "loss": loss_components,      # Dict from Metrics.compute_loss_components
            "acc": accuracy_components,   # Dict from Metrics.compute_accuracy_components
            "map": map_components,   # Dict from Metrics.compute_map_components
            "f1": f1_components,   # Dict from Metrics.compute_f1_components
            "em": em_components,   # Dict from Metrics.compute_exact_match_accuracy_components
        }
        
def main_cls(model_name: str, device: torch.device) -> None:
    # Example usage
    quant_config = None

    num_classes = 80
    # Assume these are defined based on your specific task
    ALL_LABELS = list(range(80))
    PRIVILEGED_INDICES_SET = set(list(range(20))) 
    NON_PRIVILEGED_INDICES_SET = set(ALL_LABELS) - PRIVILEGED_INDICES_SET

    # Convert sets to lists/tensors for easier indexing if needed
    PRIVILEGED_INDICES = sorted(list(PRIVILEGED_INDICES_SET))
    NON_PRIVILEGED_INDICES = sorted(list(NON_PRIVILEGED_INDICES_SET))
    print(f"Privileged Indices: {PRIVILEGED_INDICES}")
    print(f"Non-Privileged Indices: {NON_PRIVILEGED_INDICES}")

    # Initialize the model
    model1 = VisionModelForCLS(
        device=device,
        model_name=model_name,
        num_labels=num_classes,
        ref_cls_weights_path=None, # "sft_classifier_weights.pth", # Example path
        privileged_indices=PRIVILEGED_INDICES,
        non_privileged_indices=NON_PRIVILEGED_INDICES,
        loss_type="dpo", # "dpo", "simpo", "cpo" or None if is_ref is True
        beta=1.0,      # Example value
        epsilon=0.1,   # Example value
        is_ref=False,  # Set to True for reference model
        quant_config=quant_config
    ).to(device)
    model2 = VisionModelForCLS(
        device=device,
        model_name=model_name,
        num_labels=num_classes,
        ref_cls_weights_path=None, # "sft_classifier_weights.pth", # Example path
        privileged_indices=PRIVILEGED_INDICES,
        non_privileged_indices=NON_PRIVILEGED_INDICES,
        loss_type=None,
        beta=1.0,      # Example value
        epsilon=0.1,   # Example value
        is_ref=True,  # Set to True for reference model
        quant_config=quant_config
    ).to(device)

    model1.calc_num_params()
    model2.calc_num_params()

    # Dummy input data
    batch_size = 4
    dummy_pixels = torch.rand(batch_size, 3, 224, 224).to(device) # Random pixel values
    # Dummy labels (multi-label, binary)
    dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).to(device)

    print("\n--- Running Forward Pass ---")
    # Run forward pass
    output_dict1 = model1(dummy_pixels, dummy_labels)
    output_dict2 = model2(dummy_pixels, dummy_labels)

    if output_dict1["loss"] is not None:
        print("\n--- Loss Components ---")
        print(output_dict1["loss"])
        print("\n--- Accuracy Components ---")
        print(output_dict1["acc"])
        print("\n--- MAP Components ---")
        print(output_dict1["map"])
        print("\n--- F1 Components ---")
        print(output_dict1["f1"])
        print("\n--- EM Components ---")
        print(output_dict1["em"])
    else:
        print("Not computed (labels were None).")
    
    if output_dict2["loss"] is not None:
        print("\n--- Loss Components ---")
        print(output_dict2["loss"])
        print("\n--- Accuracy Components ---")
        print(output_dict2["acc"])
        print("\n--- MAP Components ---")
        print(output_dict2["map"])
        print("\n--- F1 Components ---")
        print(output_dict2["f1"])
        print("\n--- EM Components ---")
        print(output_dict2["em"])
    else:
        print("Not computed (labels were None).")
        
    print("\nDONE")

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {dev}...")
    # Using a smaller model for potentially faster testing if needed
    main_cls("google/vit-base-patch16-224", device=dev) # Or use the base model