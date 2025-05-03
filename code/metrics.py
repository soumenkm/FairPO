# metrics.py
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
import numpy as np
from typing import List, Dict, Optional, Tuple, Union

class Metrics:
    """
    A collection of static methods for calculating losses and accuracies
    relevant to the FairPO framework and multi-label classification.
    Loss and Accuracy computations are separated into distinct static methods.
    """
    eps = 1e-8 # Class attribute for stability epsilon

    # --- Internal Loss Helpers ---
    @staticmethod
    def _compute_bce_loss(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Internal helper for standard Binary Cross-Entropy loss."""
        prob_scores_clamped = prob_scores.clamp(min=Metrics.eps, max=1.0 - Metrics.eps)
        loss = F.binary_cross_entropy(
            prob_scores_clamped,
            labels.to(torch.float32),
            reduction=reduction
        )
        return loss

    @staticmethod
    def _compute_privileged_loss_dpo(
        prob_scores: torch.Tensor,
        ref_prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        num_labels: int,
        beta: float,
        device: torch.device
    ) -> torch.Tensor:
        """Internal helper for DPO-inspired privileged loss."""
        batch_size = prob_scores.shape[0]
        total_priv_loss = 0.0
        num_pairs = 0

        if not privileged_indices:
            return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)

        for i in range(batch_size):
            current_scores_i = prob_scores[i]
            ref_scores_i = ref_prob_scores[i]
            labels_i = labels[i]

            positive_priv_labels = [
                l for l in privileged_indices if 0 <= l < num_labels and labels_i[l] == 1
            ]

            for l in positive_priv_labels:
                m_il = current_scores_i[l]
                confusing_negatives = [
                    k for k in range(num_labels)
                    if labels_i[k] == 0 and current_scores_i[k] >= m_il
                ]

                for k in confusing_negatives:
                    m_ik = current_scores_i[k]
                    ref_m_il = ref_scores_i[l]
                    ref_m_ik = ref_scores_i[k]

                    log_term_l = torch.log(m_il / (ref_m_il + Metrics.eps) + Metrics.eps)
                    log_term_k = torch.log(m_ik / (ref_m_ik + Metrics.eps) + Metrics.eps)
                    h_w = log_term_l - log_term_k
                    loss_term = -F.logsigmoid(beta * h_w)

                    total_priv_loss += loss_term
                    num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)
        else:
            return total_priv_loss / num_pairs

    @staticmethod
    def _compute_non_privileged_loss(
        prob_scores: torch.Tensor,
        ref_prob_scores: torch.Tensor,
        labels: torch.Tensor,
        non_privileged_indices: List[int],
        epsilon: float,
        device: torch.device
    ) -> torch.Tensor:
        """Internal helper for constrained non-privileged loss."""
        if not non_privileged_indices:
             return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)

        non_priv_indices_tensor = torch.tensor(non_privileged_indices, device=device, dtype=torch.long)
        # Ensure indices are valid before slicing
        valid_indices = non_priv_indices_tensor[non_priv_indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(non_privileged_indices):
            print(f"Warning: {len(non_privileged_indices) - len(valid_indices)} non-privileged indices out of bounds and ignored.")
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)
            non_priv_indices_tensor = valid_indices # Use only valid indices

        m_nonpriv = prob_scores[:, non_priv_indices_tensor]
        ref_m_nonpriv = ref_prob_scores[:, non_priv_indices_tensor]
        y_nonpriv = labels[:, non_priv_indices_tensor]

        loss_current = Metrics._compute_bce_loss(m_nonpriv, y_nonpriv, reduction='none')
        with torch.no_grad():
            loss_ref = Metrics._compute_bce_loss(ref_m_nonpriv, y_nonpriv, reduction='none')

        hinge_loss = F.relu(loss_current - loss_ref - epsilon)
        avg_non_priv_loss = torch.mean(hinge_loss)
        return avg_non_priv_loss
    
    @staticmethod
    def compute_loss_components(
        prob_scores: torch.Tensor,
        ref_prob_scores: Optional[torch.Tensor],
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int],
        is_ref_mode: bool,
        beta: float,
        epsilon: float,
        loss_type: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Computes only the relevant loss components based on the mode.

        Args:
            (Same as compute_all_metrics)

        Returns:
            loss_components (Dict[str, torch.Tensor] or None): Dictionary of loss tensors.
            Returns None if labels are not provided.
        """
        if labels is None:
            return None

        device = prob_scores.device
        num_labels = prob_scores.shape[1]
        loss_components = {}

        if is_ref_mode:
            loss_components["loss"] = Metrics._compute_bce_loss(
                prob_scores=prob_scores,
                labels=labels,
                reduction='mean'
            )
        else:
            if ref_prob_scores is None:
                raise ValueError("ref_prob_scores must be provided when is_ref_mode is False.")

            if loss_type == "dpo":
                loss_components["privileged"] = Metrics._compute_privileged_loss_dpo(
                    prob_scores=prob_scores,
                    ref_prob_scores=ref_prob_scores,
                    labels=labels,
                    privileged_indices=privileged_indices,
                    num_labels=num_labels,
                    beta=beta,
                    device=device
                )
            else:
                raise NotImplementedError(f"Loss type '{loss_type}' is not implemented.")
            
            loss_components["non_privileged"] = Metrics._compute_non_privileged_loss(
                prob_scores=prob_scores,
                ref_prob_scores=ref_prob_scores,
                labels=labels,
                non_privileged_indices=non_privileged_indices,
                epsilon=epsilon,
                device=device
            )
            # Provide a 'loss' key as the raw sum for convenience during training logging/combination
            # The actual weighted combination happens in the trainer.
            loss_components["loss"] = loss_components["privileged"] + loss_components["non_privileged"]

        return loss_components

    # --- Internal Accuracy Helpers ---
    @staticmethod
    def _compute_accuracy(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """Internal helper for overall element-wise accuracy."""
        if prob_scores.numel() == 0 or labels.numel() == 0: return float('nan')
        labels_float = labels.to(torch.float32)
        predictions = (prob_scores >= threshold).to(torch.float32)
        correct = (predictions == labels_float)
        accuracy = correct.float().mean().item()
        return accuracy

    @staticmethod
    def _compute_accuracy_subset(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        indices: List[int],
        threshold: float,
        device: torch.device
    ) -> float:
        """Internal helper for subset accuracy."""
        if not indices: return float('nan')
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        # Ensure indices are valid before slicing
        valid_indices = indices_tensor[indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(indices):
             print(f"Warning: {len(indices) - len(valid_indices)} subset accuracy indices out of bounds and ignored.")
             if len(valid_indices) == 0: return float('nan')
             indices_tensor = valid_indices # Use only valid indices

        scores_subset = prob_scores[:, indices_tensor]
        labels_subset = labels[:, indices_tensor]
        accuracy = Metrics._compute_accuracy(scores_subset, labels_subset, threshold)
        return accuracy

    # --- Public Static Methods for Accuracy ---

    @staticmethod
    def compute_accuracy_components(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int],
        accuracy_threshold: float = 0.5
    ) -> Optional[Dict[str, float]]:
        """
        Computes all relevant accuracy components.

        Args:
             (Same as compute_all_metrics, minus loss-specific params like beta, epsilon)

        Returns:
            accuracy_components (Dict[str, float] or None): Dictionary of accuracy floats.
            Returns None if labels are not provided.
        """
        if labels is None:
            return None

        device = prob_scores.device
        accuracy_components = {}

        accuracy_components["acc"] = Metrics._compute_accuracy(
            prob_scores, labels, accuracy_threshold
        )
        accuracy_components["privileged"] = Metrics._compute_accuracy_subset(
            prob_scores, labels, privileged_indices, accuracy_threshold, device
        )
        accuracy_components["non_privileged"] = Metrics._compute_accuracy_subset(
            prob_scores, labels, non_privileged_indices, accuracy_threshold, device
        )

        return accuracy_components
    
    # --- Internal Exact Match Accuracy Helpers ---

    @staticmethod
    def _compute_exact_match_accuracy(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        (Internal Helper) Computes overall exact match accuracy (ratio of instances
        where predicted labels perfectly match true labels).
        """
        if prob_scores.numel() == 0 or labels.numel() == 0: return float('nan')
        if prob_scores.shape != labels.shape:
            print("Warning: Shape mismatch for exact match accuracy.")
            return float('nan')

        # Get binary predictions and ensure labels are integer type for comparison
        predictions_binary = (prob_scores >= threshold).int()
        labels_binary = labels.int()

        # Compare each row (instance) for exact match
        # torch.all(tensor, dim=1) checks if all elements along dimension 1 (columns) are True
        matches = torch.all(predictions_binary == labels_binary, dim=1)

        # Calculate the ratio of matching instances
        exact_match_ratio = matches.float().mean().item()
        return exact_match_ratio

    @staticmethod
    def _compute_exact_match_accuracy_subset(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        indices: List[int],
        threshold: float,
        device: torch.device
    ) -> float:
        """
        (Internal Helper) Computes exact match accuracy considering only a subset of labels.
        """
        if not indices: return float('nan')
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        # Ensure indices are valid before slicing
        valid_indices = indices_tensor[indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} exact match subset indices out of bounds ignored.")
            if len(valid_indices) == 0: return float('nan')
            indices_tensor = valid_indices # Use only valid indices

        # Get binary predictions and labels for the subset
        predictions_binary_subset = (prob_scores[:, indices_tensor] >= threshold).int()
        labels_binary_subset = labels[:, indices_tensor].int()

        # Compare each row (instance) within the subset for exact match
        matches_subset = torch.all(predictions_binary_subset == labels_binary_subset, dim=1)

        # Calculate the ratio of matching instances (over the subset)
        exact_match_ratio_subset = matches_subset.float().mean().item()
        return exact_match_ratio_subset

    # --- Public Static Methods for Exact Match Accuracy ---

    @staticmethod
    def compute_exact_match_accuracy_components(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int],
        accuracy_threshold: float = 0.5
    ) -> Optional[Dict[str, float]]:
        """
        Computes exact match accuracy overall, and separately for the privileged
        and non-privileged label subsets.

        Args:
            prob_scores: Predicted probabilities. Shape (b, c).
            labels: Ground truth labels. Shape (b, c). Assumed on the same device.
            privileged_indices: List of indices for group P.
            non_privileged_indices: List of indices for group P_bar.
            accuracy_threshold: Threshold for converting probabilities to binary predictions.


        Returns:
            exact_match_acc_components (Dict[str, float] or None): Dictionary containing
             'exact_match_acc', 'exact_match_acc_privileged', 'exact_match_acc_non_privileged'.
             Returns None if labels is None.
        """
        if labels is None:
            return None

        device = prob_scores.device
        exact_match_components = {}

        # Calculate overall exact match accuracy
        exact_match_components["em"] = Metrics._compute_exact_match_accuracy(
            prob_scores, labels, accuracy_threshold
        )
        # Calculate exact match accuracy considering only privileged labels
        exact_match_components["em_privileged"] = Metrics._compute_exact_match_accuracy_subset(
            prob_scores, labels, privileged_indices, accuracy_threshold, device
        )
        # Calculate exact match accuracy considering only non-privileged labels
        exact_match_components["em_non_privileged"] = Metrics._compute_exact_match_accuracy_subset(
            prob_scores, labels, non_privileged_indices, accuracy_threshold, device
        )

        return exact_match_components
    
    @staticmethod
    def _compute_ap_single_label(
        y_true_single: np.ndarray,
        y_pred_single: np.ndarray
    ) -> float:
        """
        (Helper) Computes Average Precision for a single label.

        Args:
            y_true_single: Numpy array of ground truth labels (0 or 1) for one label.
            y_pred_single: Numpy array of predicted scores/probabilities for one label.

        Returns:
            Average Precision score, or NaN if no positive labels exist in y_true_single.
        """
        if y_true_single.shape != y_pred_single.shape:
            # Log warning or raise error - consistency check
            print(f"Warning: Shape mismatch in _compute_ap_single_label ({y_true_single.shape} vs {y_pred_single.shape})")
            return np.nan
        if len(y_true_single.shape) != 1:
             print("Warning: Inputs must be 1-dimensional for single label AP.")
             return np.nan

        if np.sum(y_true_single) == 0:
            return np.nan # AP undefined if no true positives

        ap = average_precision_score(y_true_single, y_pred_single)
        return float(ap)

    @staticmethod
    def _compute_map(
        prob_scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        (Internal Helper) Computes overall Mean Average Precision (mAP) across all labels.
        Mirrors _compute_accuracy structure.

        Args:
            prob_scores: Predicted probabilities. Shape (b, c).
            labels: Ground truth labels. Shape (b, c).

        Returns:
            Overall mAP score (float). Returns NaN if mAP cannot be computed for any label.
        """
        if prob_scores.numel() == 0 or labels.numel() == 0:
            return float('nan')
        if prob_scores.shape != labels.shape:
             print("Warning: Shape mismatch between scores and labels for mAP calculation.")
             return float('nan')

        num_labels = prob_scores.shape[1]
        ap_scores = []

        y_true_np = labels.detach().cpu().numpy()
        y_pred_np = prob_scores.detach().cpu().numpy()

        for i in range(num_labels):
            ap = Metrics._compute_ap_single_label(y_true_np[:, i], y_pred_np[:, i])
            if not np.isnan(ap): # Only average valid AP scores
                ap_scores.append(ap)

        if not ap_scores:
            return float('nan')
        else:
            return float(np.mean(ap_scores))

    @staticmethod
    def _compute_map_subset(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        indices: List[int],
        device: torch.device
    ) -> float:
        """
        (Internal Helper) Computes mAP specifically for a subset of labels.
        Mirrors _compute_accuracy_subset structure.

        Args:
            prob_scores: Predicted probabilities. Shape (b, c).
            labels: Ground truth labels. Shape (b, c).
            indices: List of indices for the subset.
            device: The torch device (used for index validation).

        Returns:
            mAP score for the subset (float). Returns NaN if indices are invalid/empty
            or mAP cannot be computed for any label in the subset.
        """
        if not indices:
            return float('nan')

        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        valid_indices = indices_tensor[indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(indices):
             print(f"Warning: {len(indices) - len(valid_indices)} mAP subset indices out of bounds ignored.")
             if len(valid_indices) == 0: return float('nan')
             indices_tensor = valid_indices

        scores_subset = prob_scores[:, indices_tensor]
        labels_subset = labels[:, indices_tensor]

        # Calculate mAP using the general _compute_map method on the subset
        map_subset = Metrics._compute_map(scores_subset, labels_subset)
        return map_subset

    # --- Public Static Method for mAP Components ---

    @staticmethod
    def compute_map_components(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int]
    ) -> Optional[Dict[str, float]]:
        """
        Computes mAP overall, for the privileged subset, and for the non-privileged subset.
        Mirrors compute_accuracy_components structure.

        Args:
            prob_scores: Predicted probabilities. Shape (b, c).
            labels: Ground truth labels. Shape (b, c). Assumed on the same device.
            privileged_indices: List of indices for group P.
            non_privileged_indices: List of indices for group P_bar.

        Returns:
            map_components (Dict[str, float] or None): Dictionary containing 'mAP',
             'mAP_privileged', 'mAP_non_privileged'. Returns None if labels is None.
        """
        if labels is None:
            return None

        device = prob_scores.device # Get device from input tensor
        map_components = {}

        # Call internal helpers
        map_components["mAP"] = Metrics._compute_map(
            prob_scores, labels
        )
        map_components["mAP_privileged"] = Metrics._compute_map_subset(
            prob_scores, labels, privileged_indices, device
        )
        map_components["mAP_non_privileged"] = Metrics._compute_map_subset(
            prob_scores, labels, non_privileged_indices, device
        )

        return map_components
    
    @staticmethod
    def _compute_f1(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: float,
        average_mode: str = 'samples', # Use 'samples' for Sample-Based F1
        zero_division: Union[str, int] = 0
    ) -> float:
        """
        (Internal Helper) Computes F1 score using the specified averaging method ('samples').

        Args:
            prob_scores: Predicted probabilities. Shape (b, c).
            labels: Ground truth labels. Shape (b, c).
            threshold: Threshold for converting probabilities to binary predictions.
            average_mode: Averaging mode for f1_score ('samples').
            zero_division: Value for F1 when P+R=0.

        Returns:
            F1 score (float). Returns NaN on error or invalid input.
        """
        if prob_scores.numel() == 0 or labels.numel() == 0: return float('nan')
        if prob_scores.shape != labels.shape:
            print("Warning: Shape mismatch between scores and labels for F1 calculation.")
            return float('nan')

        # Get binary predictions
        y_pred_binary_np = (prob_scores.detach().cpu().numpy() >= threshold).astype(int)
        y_true_np = labels.detach().cpu().numpy().astype(int)

        f1 = f1_score(y_true_np, y_pred_binary_np, average=average_mode, zero_division=zero_division)
        return float(f1)

    @staticmethod
    def _compute_f1_subset(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        indices: List[int],
        threshold: float,
        device: torch.device,
        average_mode: str = 'samples', 
        zero_division: Union[str, int] = 0
    ) -> float:
        """
        (Internal Helper) Computes Sample-Based F1 score specifically for a subset of labels.
        Uses sklearn f1_score with average='samples' on the subset.

        Args:
            prob_scores: Predicted probabilities. Shape (b, c).
            labels: Ground truth labels. Shape (b, c).
            indices: List of indices defining the label subset.
            threshold: Classification threshold.
            device: The torch device (used for index validation).
            average_mode: Averaging mode ('samples').
            zero_division: Value for F1 when P+R=0.

        Returns:
            Sample-Based F1 score for the subset (float). Returns NaN if indices are invalid/empty.
        """
        if not indices: return float('nan')
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        valid_indices = indices_tensor[indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} F1 subset indices out of bounds ignored.")
            if len(valid_indices) == 0: return float('nan')
            indices_tensor = valid_indices

        # Slice the TENSORS first
        scores_subset = prob_scores[:, indices_tensor]
        labels_subset = labels[:, indices_tensor]

        # Calculate Sample-Based F1 using the general _compute_f1 method on the subset tensors
        f1_subset = Metrics._compute_f1(scores_subset, labels_subset, threshold, average_mode=average_mode, zero_division=zero_division)
        return f1_subset

    # --- Public Static Method for F1 Components ---

    @staticmethod
    def compute_f1_components(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int],
        threshold: float = 0.5
    ) -> Optional[Dict[str, float]]:
        """
        Computes Sample-Based F1 score (overall average instance F1) and the
        average instance F1 calculated over privileged and non-privileged label subsets,
        using sklearn.metrics.f1_score(average='samples').

        Args:
            prob_scores: Predicted probabilities. Shape (b, c).
            labels: Ground truth labels. Shape (b, c). Assumed on the same device.
            privileged_indices: List of indices for group P.
            non_privileged_indices: List of indices for group P_bar.
            threshold: Threshold for converting probabilities to binary predictions.

        Returns:
            f1_components (Dict[str, float] or None): Dictionary containing
             'f1' (overall sample-based F1), 'f1_privileged', 'f1_non_privileged'.
             Returns None if labels is None.
        """
        if labels is None:
            return None

        device = prob_scores.device # Get device from input tensor
        f1_components = {}
        avg_mode = 'samples' # Specify sample-based averaging

        # Call internal helpers for sample-based F1
        f1_components["f1"] = Metrics._compute_f1(
            prob_scores, labels, threshold, average_mode=avg_mode
        )
        f1_components["f1_privileged"] = Metrics._compute_f1_subset(
            prob_scores, labels, privileged_indices, threshold, device, average_mode=avg_mode
        )
        f1_components["f1_non_privileged"] = Metrics._compute_f1_subset(
            prob_scores, labels, non_privileged_indices, threshold, device, average_mode=avg_mode
        )

        return f1_components