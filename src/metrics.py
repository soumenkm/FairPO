import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import random

class Metrics:
    """
    A collection of static methods for calculating losses and accuracies
    relevant to the FairPO framework and multi-label classification.
    Loss and accuracy computations are separated into distinct static methods.
    """
    eps = 1e-8

    @staticmethod
    def _compute_bce_loss(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Computes standard Binary Cross-Entropy loss with clamping for numerical stability.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            reduction (str): Reduction method ('mean', 'sum', or 'none').

        Returns:
            torch.Tensor: BCE loss tensor.
        """
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
        """
        Computes the DPO-inspired privileged loss by comparing positive privileged labels
        against confusing negatives using reference and current model scores.

        Args:
            prob_scores (torch.Tensor): Current model probabilities (batch_size, num_labels).
            ref_prob_scores (torch.Tensor): Reference model probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): List of privileged label indices.
            num_labels (int): Total number of labels.
            beta (float): Scaling factor.
            device (torch.device): Torch device.

        Returns:
            torch.Tensor: Averaged privileged loss scalar tensor.
        """
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
    def _compute_privileged_loss_simpo(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        num_labels: int,
        beta: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Computes the SimPO-inspired privileged loss without reference scores,
        comparing positive privileged labels against confusing negatives.

        Args:
            prob_scores (torch.Tensor): Current model probabilities (batch_size, num_labels).
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): List of privileged label indices.
            num_labels (int): Total number of labels.
            beta (float): Scaling factor.
            device (torch.device): Torch device.

        Returns:
            torch.Tensor: Averaged privileged loss scalar tensor.
        """
        batch_size = prob_scores.shape[0]
        total_priv_loss = 0.0
        num_pairs = 0
        gamma = 0.0

        if not privileged_indices:
            return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)

        for i in range(batch_size):
            current_scores_i = prob_scores[i]
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

                    log_term_l = torch.log(m_il + Metrics.eps)
                    log_term_k = torch.log(m_ik + Metrics.eps)
                    h_w = log_term_l - log_term_k
                    loss_term = -F.logsigmoid(beta * h_w - gamma)

                    total_priv_loss += loss_term
                    num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)
        else:
            return total_priv_loss / num_pairs
        
    @staticmethod
    def _compute_privileged_loss_dpo_conditional(
        prob_scores: torch.Tensor,
        ref_prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        num_labels: int,
        beta: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Computes conditional DPO privileged loss applying DPO loss if confusing examples
        exist, else falls back to BCE loss for each privileged label.

        Args:
            prob_scores (torch.Tensor): Current model probabilities.
            ref_prob_scores (torch.Tensor): Reference model probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): Privileged label indices.
            num_labels (int): Total labels count.
            beta (float): Scaling factor.
            device (torch.device): Torch device.

        Returns:
            torch.Tensor: Averaged privileged loss tensor.
        """
        batch_size = prob_scores.shape[0]
        total_loss_for_privileged_labels = 0.0
        num_privileged_labels_processed = 0

        if not privileged_indices:
            return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)

        for i in range(batch_size):
            current_scores_i = prob_scores[i]
            ref_scores_i = ref_prob_scores[i]
            labels_i = labels[i]

            for l_idx in privileged_indices:
                if not (0 <= l_idx < num_labels):
                    continue 
                
                num_privileged_labels_processed += 1
                y_il = labels_i[l_idx]
                m_il_current = current_scores_i[l_idx]

                confusing_set_k_indices = []
                is_positive_case = False

                if y_il == 1:
                    is_positive_case = True
                    confusing_set_k_indices = [
                        k for k in range(num_labels)
                        if labels_i[k] == 0 and current_scores_i[k] >= m_il_current and k != l_idx
                    ]
                else:
                    confusing_set_k_indices = [
                        k for k in range(num_labels)
                        if labels_i[k] == 1 and current_scores_i[k] <= m_il_current and k != l_idx
                    ]
                
                if confusing_set_k_indices:
                    k_sampled_idx = random.choice(confusing_set_k_indices)
                    m_ik_current = current_scores_i[k_sampled_idx]
                    
                    m_il_current_clamped = m_il_current.clamp(min=Metrics.eps, max=1.0 - Metrics.eps)
                    m_ik_current_clamped = m_ik_current.clamp(min=Metrics.eps, max=1.0 - Metrics.eps)
                    ref_m_il_clamped = ref_scores_i[l_idx].clamp(min=Metrics.eps, max=1.0 - Metrics.eps)
                    ref_m_ik_clamped = ref_scores_i[k_sampled_idx].clamp(min=Metrics.eps, max=1.0 - Metrics.eps)

                    if is_positive_case:
                        log_term_l = torch.log(m_il_current_clamped / ref_m_il_clamped)
                        log_term_k = torch.log(m_ik_current_clamped / ref_m_ik_clamped)
                        h_val = log_term_l - log_term_k
                    else:
                        log_term_k_prime = torch.log(m_ik_current_clamped / ref_m_ik_clamped)
                        log_term_l_prime = torch.log(m_il_current_clamped / ref_m_il_clamped)
                        h_val = log_term_k_prime - log_term_l_prime
                    
                    loss_pref = -F.logsigmoid(beta * h_val)
                    total_loss_for_privileged_labels += loss_pref
                else:
                    score_l = current_scores_i[l_idx].unsqueeze(0)
                    label_l = labels_i[l_idx].unsqueeze(0).to(torch.float32)
                    bce_loss_l = Metrics._compute_bce_loss(score_l, label_l, reduction='sum')
                    total_loss_for_privileged_labels += bce_loss_l
        
        if num_privileged_labels_processed == 0:
            return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)
        else:
            return total_loss_for_privileged_labels / num_privileged_labels_processed

    @staticmethod
    def _compute_privileged_loss_simpo_conditional(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        num_labels: int,
        beta: float,
        gamma_simpo: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Computes conditional SimPO privileged loss applying SimPO loss if confusing
        examples exist, else uses BCE loss for privileged labels.

        Args:
            prob_scores (torch.Tensor): Current model probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): Privileged label indices.
            num_labels (int): Total labels count.
            beta (float): Scaling factor.
            gamma_simpo (float): Margin parameter.
            device (torch.device): Torch device.

        Returns:
            torch.Tensor: Averaged privileged loss tensor.
        """
        batch_size = prob_scores.shape[0]
        total_loss_for_privileged_labels = 0.0
        num_privileged_labels_processed = 0

        if not privileged_indices:
            return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)

        for i in range(batch_size):
            current_scores_i = prob_scores[i]
            labels_i = labels[i]

            for l_idx in privileged_indices:
                if not (0 <= l_idx < num_labels): continue
                num_privileged_labels_processed += 1
                y_il = labels_i[l_idx]
                m_il_current = current_scores_i[l_idx]

                confusing_set_k_indices = []
                is_positive_case = False

                if y_il == 1:
                    is_positive_case = True
                    confusing_set_k_indices = [
                        k for k in range(num_labels)
                        if labels_i[k] == 0 and current_scores_i[k] >= m_il_current and k != l_idx
                    ]
                else:
                    confusing_set_k_indices = [
                        k for k in range(num_labels)
                        if labels_i[k] == 1 and current_scores_i[k] <= m_il_current and k != l_idx
                    ]
                
                if confusing_set_k_indices:
                    k_sampled_idx = random.choice(confusing_set_k_indices)
                    m_ik_current = current_scores_i[k_sampled_idx]

                    m_il_clamped = m_il_current.clamp(min=Metrics.eps)
                    m_ik_clamped = m_ik_current.clamp(min=Metrics.eps)

                    if is_positive_case:
                        log_score_preferred = torch.log(m_il_clamped)
                        log_score_dispreferred = torch.log(m_ik_clamped)
                    else:
                        log_score_preferred = torch.log(m_ik_clamped)
                        log_score_dispreferred = torch.log(m_il_clamped)
                    
                    h_val = log_score_preferred - log_score_dispreferred
                    loss_pref = -F.logsigmoid(beta * h_val - gamma_simpo)
                    total_loss_for_privileged_labels += loss_pref
                else:
                    score_l = current_scores_i[l_idx].unsqueeze(0)
                    label_l = labels_i[l_idx].unsqueeze(0).to(torch.float32)
                    bce_loss_l = Metrics._compute_bce_loss(score_l, label_l, reduction='sum')
                    total_loss_for_privileged_labels += bce_loss_l
        
        if num_privileged_labels_processed == 0:
            return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)
        else:
            return total_loss_for_privileged_labels / num_privileged_labels_processed

    @staticmethod
    def _compute_privileged_loss_cpo_conditional(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        num_labels: int,
        beta: float,
        lambda_cpo_nll: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Computes conditional CPO privileged loss with preference term applied if confusing
        examples exist and unconditional BCE applied to all privileged labels.

        Args:
            prob_scores (torch.Tensor): Current model probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): Privileged label indices.
            num_labels (int): Total labels count.
            beta (float): Scaling factor.
            lambda_cpo_nll (float): Weighting factor for NLL component.
            device (torch.device): Torch device.

        Returns:
            torch.Tensor: Final combined CPO loss.
        """
        batch_size = prob_scores.shape[0]
        total_preference_loss_sum = 0.0
        num_preference_pairs_processed_for_avg = 0

        if privileged_indices:
            for i in range(batch_size):
                current_scores_i = prob_scores[i]
                labels_i = labels[i]

                for l_idx in privileged_indices:
                    if not (0 <= l_idx < num_labels): continue
                    
                    y_il = labels_i[l_idx]
                    m_il_current = current_scores_i[l_idx]
                    confusing_set_k_indices = []
                    is_positive_case_for_pref = False

                    if y_il == 1:
                        is_positive_case_for_pref = True
                        confusing_set_k_indices = [
                            k for k in range(num_labels)
                            if labels_i[k] == 0 and current_scores_i[k] >= m_il_current and k != l_idx
                        ]
                    else:
                        confusing_set_k_indices = [
                            k for k in range(num_labels)
                            if labels_i[k] == 1 and current_scores_i[k] <= m_il_current and k != l_idx
                        ]

                    if confusing_set_k_indices:
                        k_sampled_idx = random.choice(confusing_set_k_indices)
                        m_ik_current = current_scores_i[k_sampled_idx]

                        m_il_clamped = m_il_current.clamp(min=Metrics.eps)
                        m_ik_clamped = m_ik_current.clamp(min=Metrics.eps)

                        if is_positive_case_for_pref:
                            log_score_preferred = torch.log(m_il_clamped)
                            log_score_dispreferred = torch.log(m_ik_clamped)
                        else:
                            log_score_preferred = torch.log(m_ik_clamped)
                            log_score_dispreferred = torch.log(m_il_clamped)
                        
                        h_val = log_score_preferred - log_score_dispreferred
                        preference_loss_term = -F.logsigmoid(beta * h_val)
                        total_preference_loss_sum += preference_loss_term
                        num_preference_pairs_processed_for_avg += 1
            
        avg_preference_loss = (total_preference_loss_sum / num_preference_pairs_processed_for_avg) \
                              if num_preference_pairs_processed_for_avg > 0 \
                              else torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)

        avg_nll_loss_for_privileged = torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)
        if privileged_indices:
            priv_indices_tensor_for_nll = torch.tensor(
                [idx for idx in privileged_indices if 0 <= idx < num_labels], 
                device=device, dtype=torch.long
            )
            if priv_indices_tensor_for_nll.numel() > 0:
                scores_privileged_subset = prob_scores[:, priv_indices_tensor_for_nll]
                labels_privileged_subset = labels[:, priv_indices_tensor_for_nll]
                avg_nll_loss_for_privileged = Metrics._compute_bce_loss(
                    scores_privileged_subset, 
                    labels_privileged_subset, 
                    reduction='mean' 
                )
        
        final_cpo_loss = avg_preference_loss + lambda_cpo_nll * avg_nll_loss_for_privileged
        
        return final_cpo_loss
    
    @staticmethod
    def _compute_non_privileged_loss(
        prob_scores: torch.Tensor,
        ref_prob_scores: torch.Tensor,
        labels: torch.Tensor,
        non_privileged_indices: List[int],
        epsilon: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Computes constrained non-privileged loss as hinge loss between current and
        reference BCE losses.

        Args:
            prob_scores (torch.Tensor): Current model probabilities.
            ref_prob_scores (torch.Tensor): Reference model probabilities.
            labels (torch.Tensor): Ground truth labels.
            non_privileged_indices (List[int]): Non-privileged label indices.
            epsilon (float): Margin parameter.
            device (torch.device): Torch device.

        Returns:
            torch.Tensor: Averaged hinge loss scalar tensor.
        """
        if not non_privileged_indices:
             return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)

        non_priv_indices_tensor = torch.tensor(non_privileged_indices, device=device, dtype=torch.long)
        valid_indices = non_priv_indices_tensor[non_priv_indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(non_privileged_indices):
            print(f"Warning: {len(non_privileged_indices) - len(valid_indices)} non-privileged indices out of bounds and ignored.")
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=device, requires_grad=prob_scores.requires_grad)
            non_priv_indices_tensor = valid_indices

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
        Computes loss components based on training mode and loss type.

        Args:
            prob_scores (torch.Tensor): Current model probabilities.
            ref_prob_scores (Optional[torch.Tensor]): Reference model probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): Privileged label indices.
            non_privileged_indices (List[int]): Non-privileged label indices.
            is_ref_mode (bool): Whether in reference mode (only BCE loss).
            beta (float): Scaling factor for privileged losses.
            epsilon (float): Margin for constrained losses.
            loss_type (str): One of 'dpo', 'simpo', 'cpo'.

        Returns:
            Optional[Dict[str, torch.Tensor]]: Dictionary with loss components or None if labels are None.
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
                loss_components["privileged"] = Metrics._compute_privileged_loss_dpo_conditional(
                    prob_scores=prob_scores,
                    ref_prob_scores=ref_prob_scores,
                    labels=labels,
                    privileged_indices=privileged_indices,
                    num_labels=num_labels,
                    beta=beta,
                    device=device
                )
            elif loss_type == "simpo":
                loss_components["privileged"] = Metrics._compute_privileged_loss_simpo_conditional(
                    prob_scores=prob_scores,
                    labels=labels,
                    privileged_indices=privileged_indices,
                    num_labels=num_labels,
                    beta=beta,
                    device=device
                )
            elif loss_type == "cpo":
                loss_components["privileged"] = Metrics._compute_privileged_loss_cpo_conditional(
                    prob_scores=prob_scores,
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
            loss_components["loss"] = loss_components["privileged"] + loss_components["non_privileged"]

        return loss_components

    @staticmethod
    def _compute_accuracy(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Computes overall element-wise accuracy for multi-label classification.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            threshold (float): Threshold to convert probabilities to binary predictions.

        Returns:
            float: Accuracy score.
        """
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
        """
        Computes element-wise accuracy restricted to a subset of labels.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            indices (List[int]): Label indices to compute accuracy on.
            threshold (float): Threshold to convert probabilities to binary predictions.
            device (torch.device): Torch device.

        Returns:
            float: Accuracy score on subset.
        """
        if not indices: return float('nan')
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        valid_indices = indices_tensor[indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} subset accuracy indices out of bounds and ignored.")
            if len(valid_indices) == 0: return float('nan')
            indices_tensor = valid_indices

        scores_subset = prob_scores[:, indices_tensor]
        labels_subset = labels[:, indices_tensor]
        accuracy = Metrics._compute_accuracy(scores_subset, labels_subset, threshold)
        return accuracy

    @staticmethod
    def compute_accuracy_components(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int],
        accuracy_threshold: float = 0.5
    ) -> Optional[Dict[str, float]]:
        """
        Computes overall, privileged, and non-privileged accuracy components.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): Privileged label indices.
            non_privileged_indices (List[int]): Non-privileged label indices.
            accuracy_threshold (float): Threshold for binary predictions.

        Returns:
            Optional[Dict[str, float]]: Dictionary of accuracy components or None if labels are None.
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

    @staticmethod
    def _compute_exact_match_accuracy(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Computes exact match accuracy, where predicted labels exactly match true labels.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            threshold (float): Threshold for binary predictions.

        Returns:
            float: Exact match accuracy.
        """
        if prob_scores.numel() == 0 or labels.numel() == 0: return float('nan')
        if prob_scores.shape != labels.shape:
            print("Warning: Shape mismatch for exact match accuracy.")
            return float('nan')

        predictions_binary = (prob_scores >= threshold).int()
        labels_binary = labels.int()

        matches = torch.all(predictions_binary == labels_binary, dim=1)
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
        Computes exact match accuracy considering only a subset of labels.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            indices (List[int]): Label indices for subset.
            threshold (float): Threshold for binary predictions.
            device (torch.device): Torch device.

        Returns:
            float: Exact match accuracy on subset.
        """
        if not indices: return float('nan')
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        valid_indices = indices_tensor[indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} exact match subset indices out of bounds ignored.")
            if len(valid_indices) == 0: return float('nan')
            indices_tensor = valid_indices

        predictions_binary_subset = (prob_scores[:, indices_tensor] >= threshold).int()
        labels_binary_subset = labels[:, indices_tensor].int()

        matches_subset = torch.all(predictions_binary_subset == labels_binary_subset, dim=1)
        exact_match_ratio_subset = matches_subset.float().mean().item()
        return exact_match_ratio_subset

    @staticmethod
    def compute_exact_match_accuracy_components(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int],
        accuracy_threshold: float = 0.5
    ) -> Optional[Dict[str, float]]:
        """
        Computes exact match accuracy overall, privileged subset, and non-privileged subset.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): Privileged label indices.
            non_privileged_indices (List[int]): Non-privileged label indices.
            accuracy_threshold (float): Threshold for binary predictions.

        Returns:
            Optional[Dict[str, float]]: Dictionary with exact match accuracy components or None.
        """
        if labels is None:
            return None

        device = prob_scores.device
        exact_match_components = {}

        exact_match_components["em"] = Metrics._compute_exact_match_accuracy(
            prob_scores, labels, accuracy_threshold
        )
        exact_match_components["em_privileged"] = Metrics._compute_exact_match_accuracy_subset(
            prob_scores, labels, privileged_indices, accuracy_threshold, device
        )
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
        Computes Average Precision (AP) for a single label.

        Args:
            y_true_single (np.ndarray): Ground truth binary labels for a single class.
            y_pred_single (np.ndarray): Predicted scores for the same class.

        Returns:
            float: Average precision score or NaN if no positives.
        """
        if y_true_single.shape != y_pred_single.shape:
            print(f"Warning: Shape mismatch in _compute_ap_single_label ({y_true_single.shape} vs {y_pred_single.shape})")
            return np.nan
        if len(y_true_single.shape) != 1:
             print("Warning: Inputs must be 1-dimensional for single label AP.")
             return np.nan

        if np.sum(y_true_single) == 0:
            return np.nan

        ap = average_precision_score(y_true_single, y_pred_single)
        return float(ap)

    @staticmethod
    def _compute_map(
        prob_scores: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Computes mean Average Precision (mAP) across all labels.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            float: Mean Average Precision.
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
            if not np.isnan(ap):
                ap_scores.append(ap)

        if not ap_scores:
            return 0.0
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
        Computes mAP for a subset of labels.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            indices (List[int]): Indices of labels for the subset.
            device (torch.device): Torch device.

        Returns:
            float: mAP score for the subset.
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

        map_subset = Metrics._compute_map(scores_subset, labels_subset)
        return map_subset

    @staticmethod
    def compute_map_components(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int]
    ) -> Optional[Dict[str, float]]:
        """
        Computes mAP overall, privileged subset, and non-privileged subset.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): Privileged label indices.
            non_privileged_indices (List[int]): Non-privileged label indices.

        Returns:
            Optional[Dict[str, float]]: Dictionary of mAP components or None.
        """
        if labels is None:
            return None

        device = prob_scores.device
        map_components = {}

        map_components["mAP"] = Metrics._compute_map(prob_scores, labels)
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
        average_mode: str = 'samples',
        zero_division: Union[str, int] = 0
    ) -> float:
        """
        Computes sample-based F1 score.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            threshold (float): Threshold to binarize predictions.
            average_mode (str): Averaging mode for sklearn f1_score.
            zero_division (str or int): Value to use when there is zero division.

        Returns:
            float: F1 score.
        """
        if prob_scores.numel() == 0 or labels.numel() == 0: return float('nan')
        if prob_scores.shape != labels.shape:
            print("Warning: Shape mismatch between scores and labels for F1 calculation.")
            return float('nan')

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
        Computes sample-based F1 score on a subset of labels.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            indices (List[int]): Label indices subset.
            threshold (float): Threshold to binarize predictions.
            device (torch.device): Torch device.
            average_mode (str): Averaging mode for sklearn f1_score.
            zero_division (str or int): Value to use when there is zero division.

        Returns:
            float: F1 score on subset.
        """
        if not indices: return float('nan')
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        valid_indices = indices_tensor[indices_tensor < prob_scores.shape[1]]
        if len(valid_indices) != len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} F1 subset indices out of bounds ignored.")
            if len(valid_indices) == 0: return float('nan')
            indices_tensor = valid_indices

        scores_subset = prob_scores[:, indices_tensor]
        labels_subset = labels[:, indices_tensor]

        f1_subset = Metrics._compute_f1(scores_subset, labels_subset, threshold, average_mode=average_mode, zero_division=zero_division)
        return f1_subset

    @staticmethod
    def compute_f1_components(
        prob_scores: torch.Tensor,
        labels: torch.Tensor,
        privileged_indices: List[int],
        non_privileged_indices: List[int],
        threshold: float = 0.5
    ) -> Optional[Dict[str, float]]:
        """
        Computes overall, privileged, and non-privileged sample-based F1 scores.

        Args:
            prob_scores (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.
            privileged_indices (List[int]): Privileged label indices.
            non_privileged_indices (List[int]): Non-privileged label indices.
            threshold (float): Threshold for binary predictions.

        Returns:
            Optional[Dict[str, float]]: Dictionary with F1 components or None if labels are None.
        """
        if labels is None:
            return None

        device = prob_scores.device
        f1_components = {}
        avg_mode = 'samples'

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
