import os
import sys
from pathlib import Path
from typing import Union, List, Set
import copy

import torchinfo
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForImageClassification,
                          BitsAndBytesConfig)
from metrics import Metrics

seed = 42
torch.manual_seed(seed)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class VisionModelForCLS(torch.nn.Module):
    """
    Vision Transformer based multi-label classification model with
    support for reference and trainable classifier heads, and
    privileged/non-privileged label handling with custom loss types.

    Attributes:
        device (torch.device): Device to run model on.
        model_name (str): Huggingface model checkpoint name.
        num_labels (int): Number of classification labels.
        ref_cls_wt_path (str): Path to reference classifier weights.
        privileged_indices (List[int]): Indices of privileged labels.
        non_privileged_indices (List[int]): Indices of non-privileged labels.
        is_ref (bool): Whether this is a reference model.
        loss_type (str): Loss variant ("dpo", "simpo", "cpo", or None).
        beta (float): Hyperparameter for scaling privileged loss.
        epsilon (float): Slack parameter for constraint losses.
        quant_config (BitsAndBytesConfig or None): Optional quantization config.
    """
    def __init__(self,
                 device: torch.device,
                 model_name: str,
                 num_labels: int,
                 ref_cls_weights_path: str,
                 privileged_indices: List[int],
                 non_privileged_indices: List[int],
                 is_ref: bool,
                 loss_type: str,
                 beta: float = 1.0,
                 epsilon: float = 0.1,
                 quant_config: Union[BitsAndBytesConfig, None] = None):
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
        self.eps = 1e-8
        self.loss_type = loss_type

        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            quantization_config=self.quant_config,
            device_map="auto",
        )
        self.d = self.model.config.hidden_size
        self.c = num_labels
        self.config = self.model.config
        
        self.model.classifier = self._get_classifier_head(self.d, self.c).to(self.device)
        
        if not self.is_ref:
            self.model.ref_classifier = self._get_classifier_head(self.d, self.c).to(self.device)
            if self.ref_cls_wt_path is not None:
                ref_cls_state_dict = torch.load(self.ref_cls_wt_path, map_location=self.device, weights_only=False)
                ref_cls_state_dict_new = {}
                for k, v in ref_cls_state_dict["model_state_dict"].items():
                    if "model.classifier." in k:
                        k = k.replace("model.classifier.", "")
                        ref_cls_state_dict_new[k] = v
                self.model.ref_classifier.load_state_dict(ref_cls_state_dict_new, strict=True) 
                print(f"INFO: Reference classifier weights loaded from {self.ref_cls_wt_path}")
            else:
                print("WARNING: Reference classifier weights are NOT loaded. Using initialized weights.")
            
            for param in self.model.ref_classifier.parameters():
                param.requires_grad = False

        for param in self.model.vit.parameters():
            param.requires_grad = False
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
        if not self.is_ref:
            for param in self.model.vit.encoder.layer[-1].parameters():
                param.requires_grad = True

    def _get_classifier_head(self, d: int, num_labels: int) -> torch.nn.Module:
        """
        Creates a ModuleList of MLP classifier heads, one per label.

        Args:
            d (int): Input hidden size dimension.
            num_labels (int): Number of output labels.

        Returns:
            torch.nn.ModuleList: ModuleList of MLPs for classification.
        """
        mlp_list = []
        for _ in range(num_labels):
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
        """
        Computes logits from classifier heads and returns sigmoid probabilities.

        Args:
            classifier_head (nn.ModuleList): Classifier heads.
            hidden_state (torch.tensor): Hidden state from backbone (b, d).

        Returns:
            torch.tensor: Probability scores for each label (b, c).
        """
        batch_size = hidden_state.shape[0]
        logits = torch.zeros((batch_size, self.c), device=self.device)
        for i, layer in enumerate(classifier_head):
            logits[:, i] = layer(hidden_state).squeeze(-1)
        prob_scores = torch.sigmoid(logits)
        return prob_scores

    def calc_num_params(self) -> None:
        """
        Prints the total and trainable parameters of the model.
        """
        train_params = 0
        total_params = 0
        for _, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        
        print(self)
        print(f"Backbone: {self.model_name}")
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters (classifier heads): {train_params}")
        if total_params > 0:
           print(f"Training Percentage: {train_params * 100 / total_params:.3f}%")
        else:
           print("Training Percentage: N/A (total_params is zero)")

    def forward(self, pixels: torch.tensor, labels: Union[torch.tensor, None]) -> dict:
        """
        Forward pass: compute predictions and losses for input images.

        Args:
            pixels (torch.tensor): Input images (b, 3, H, W).
            labels (torch.tensor or None): Multi-label ground truth (b, c).

        Returns:
            dict: Dictionary with keys 'outputs', 'loss', 'acc', 'map', 'f1', 'em'.
        """
        pixels = pixels.to(self.device)
        with torch.no_grad():
            outputs = self.model.vit(pixel_values=pixels, output_hidden_states=False)
            hidden_state = outputs.last_hidden_state[:, 0, :]
        hidden_state = hidden_state.to(torch.float32)

        prob_scores = self._get_scores(self.model.classifier, hidden_state)

        ref_prob_scores = None
        if not self.is_ref:
            with torch.no_grad():
                ref_prob_scores = self._get_scores(self.model.ref_classifier, hidden_state)

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
        
        if self.training and labels is None:
            raise ValueError("Labels must be provided during training.")

        return {
            "outputs": prob_scores,
            "loss": loss_components,
            "acc": accuracy_components,
            "map": map_components,
            "f1": f1_components,
            "em": em_components,
        }
        
def main_cls(model_name: str, device: torch.device) -> None:
    """
    Example usage of VisionModelForCLS with dummy data.

    Args:
        model_name (str): Huggingface model checkpoint name.
        device (torch.device): Device to run model on.
    """
    quant_config = None

    num_classes = 80
    ALL_LABELS = list(range(80))
    PRIVILEGED_INDICES_SET = set(range(20))
    NON_PRIVILEGED_INDICES_SET = set(ALL_LABELS) - PRIVILEGED_INDICES_SET

    PRIVILEGED_INDICES = sorted(list(PRIVILEGED_INDICES_SET))
    NON_PRIVILEGED_INDICES = sorted(list(NON_PRIVILEGED_INDICES_SET))
    print(f"Privileged Indices: {PRIVILEGED_INDICES}")
    print(f"Non-Privileged Indices: {NON_PRIVILEGED_INDICES}")

    model1 = VisionModelForCLS(
        device=device,
        model_name=model_name,
        num_labels=num_classes,
        ref_cls_weights_path=None,
        privileged_indices=PRIVILEGED_INDICES,
        non_privileged_indices=NON_PRIVILEGED_INDICES,
        loss_type="dpo",
        beta=1.0,
        epsilon=0.1,
        is_ref=False,
        quant_config=quant_config
    ).to(device)

    model2 = VisionModelForCLS(
        device=device,
        model_name=model_name,
        num_labels=num_classes,
        ref_cls_weights_path=None,
        privileged_indices=PRIVILEGED_INDICES,
        non_privileged_indices=NON_PRIVILEGED_INDICES,
        loss_type=None,
        beta=1.0,
        epsilon=0.1,
        is_ref=True,
        quant_config=quant_config
    ).to(device)

    model1.calc_num_params()
    model2.calc_num_params()

    batch_size = 4
    dummy_pixels = torch.rand(batch_size, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).to(device)

    print("\n--- Running Forward Pass ---")
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
    main_cls("google/vit-base-patch16-224", device=dev)
