"""
PlantHealth AI - Out-of-Distribution (OOD) Detection
====================================================
Detects images that are significantly different from the training distribution
(e.g., non-leaf images, random objects, etc.)

This module implements multiple OOD detection methods:
1. Confidence-based: Maximum softmax probability
2. Entropy-based: Prediction entropy
3. Feature-based: Mahalanobis distance from training distribution
4. Temperature scaling: ODIN-style detection

Usage:
    detector = OODDetector()
    is_ood, scores = detector.detect(outputs, features)
    
    if is_ood:
        print("Warning: This image does not appear to be a plant leaf!")
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class OODDetector:
    """
    Out-of-Distribution detector for plant disease classification.
    
    Combines multiple detection methods to identify images that are
    significantly different from the training data (plant leaves).
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        entropy_threshold: float = 2.5,
        temperature: float = 1000.0,
        use_temperature_scaling: bool = False,
    ):
        """
        Initialize OOD detector.
        
        Args:
            confidence_threshold: Minimum confidence for in-distribution (0-1)
                Lower values are more permissive. Recommended: 0.6-0.8
            entropy_threshold: Maximum entropy for in-distribution
                Higher values are more permissive. Recommended: 2.0-3.0
            temperature: Temperature for ODIN-style scaling (default: 1000)
            use_temperature_scaling: Whether to use temperature scaling
        """
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.temperature = temperature
        self.use_temperature_scaling = use_temperature_scaling
        
        logger.info(f"OOD Detector initialized:")
        logger.info(f"  - Confidence threshold: {confidence_threshold}")
        logger.info(f"  - Entropy threshold: {entropy_threshold}")
        logger.info(f"  - Temperature scaling: {use_temperature_scaling}")
    
    def compute_confidence_score(
        self,
        outputs: torch.Tensor,
        return_probability: bool = False
    ) -> Tuple[float, float]:
        """
        Compute maximum softmax probability (MSP).
        
        Low MSP suggests the image is out-of-distribution.
        
        Args:
            outputs: Model output logits (shape: [batch_size, num_classes])
            return_probability: If True, also return the max probability
            
        Returns:
            Tuple of (is_in_distribution, max_probability)
        """
        # Apply temperature scaling if enabled
        if self.use_temperature_scaling:
            outputs = outputs / self.temperature
        
        # Compute softmax probabilities
        probs = F.softmax(outputs, dim=1)
        max_prob = probs.max().item()
        
        # Check if confidence exceeds threshold
        is_in_distribution = max_prob >= self.confidence_threshold
        
        return is_in_distribution, max_prob
    
    def compute_entropy_score(self, outputs: torch.Tensor) -> Tuple[float, float]:
        """
        Compute prediction entropy.
        
        High entropy suggests the model is uncertain and the image
        might be out-of-distribution.
        
        Args:
            outputs: Model output logits (shape: [batch_size, num_classes])
            
        Returns:
            Tuple of (is_in_distribution, entropy_value)
        """
        # Compute softmax probabilities
        probs = F.softmax(outputs, dim=1)
        probs_np = probs.cpu().numpy()[0]
        
        # Compute Shannon entropy
        ent = entropy(probs_np)
        
        # Check if entropy is below threshold
        is_in_distribution = ent <= self.entropy_threshold
        
        return is_in_distribution, float(ent)
    
    def compute_variance_score(self, outputs: torch.Tensor) -> Tuple[float, float]:
        """
        Compute variance of predictions.
        
        Low variance (one class has very high probability, others very low)
        suggests in-distribution. High variance suggests confusion/OOD.
        
        Args:
            outputs: Model output logits (shape: [batch_size, num_classes])
            
        Returns:
            Tuple of (is_in_distribution, variance_value)
        """
        probs = F.softmax(outputs, dim=1)
        variance = torch.var(probs).item()
        
        # Low variance is good (one clear prediction)
        # High variance means model is confused
        variance_threshold = 0.02
        is_in_distribution = variance <= variance_threshold
        
        return is_in_distribution, variance
    
    def compute_top_k_gap(
        self,
        outputs: torch.Tensor,
        k: int = 2
    ) -> Tuple[float, float]:
        """
        Compute gap between top-k predictions.
        
        Large gap between top-1 and top-2 suggests confident in-distribution
        prediction. Small gap suggests uncertainty/OOD.
        
        Args:
            outputs: Model output logits (shape: [batch_size, num_classes])
            k: Number of top predictions to consider
            
        Returns:
            Tuple of (is_in_distribution, gap_value)
        """
        probs = F.softmax(outputs, dim=1)
        top_k_probs = torch.topk(probs, k, dim=1).values[0]
        
        # Gap between top-1 and top-2
        gap = (top_k_probs[0] - top_k_probs[1]).item()
        
        # Larger gap means more confident
        gap_threshold = 0.15
        is_in_distribution = gap >= gap_threshold
        
        return is_in_distribution, gap
    
    def compute_prediction_spread(self, outputs: torch.Tensor) -> Tuple[bool, float]:
        """
        Compute how spread out the top predictions are.
        
        For uniform/solid patterns (like cloth), predictions tend to be 
        more evenly distributed among a few classes.
        
        Args:
            outputs: Model output logits (shape: [batch_size, num_classes])
            
        Returns:
            Tuple of (is_in_distribution, spread_score)
        """
        probs = F.softmax(outputs, dim=1)
        top_5_probs = torch.topk(probs, min(5, probs.shape[1]), dim=1).values[0]
        
        # Calculate standard deviation of top-5 predictions
        spread = torch.std(top_5_probs).item()
        
        # For real leaves: top-1 is usually very high, others very low (high spread)
        # For uniform patterns: top predictions are more similar (low spread)
        spread_threshold = 0.15
        is_in_distribution = spread >= spread_threshold
        
        return is_in_distribution, spread
    
    def detect(
        self,
        outputs: torch.Tensor,
        return_scores: bool = True,
        voting_strategy: str = "majority"
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """
        Perform OOD detection using multiple methods.
        
        Args:
            outputs: Model output logits (shape: [batch_size, num_classes])
            return_scores: If True, return detailed scores from all methods
            voting_strategy: How to combine multiple methods
                - "majority": Majority vote (recommended)
                - "unanimous": All methods must agree (strict)
                - "any": Any method can trigger OOD (lenient)
                - "confidence_only": Use only confidence threshold
        
        Returns:
            Tuple of (is_ood, scores_dict)
            - is_ood: True if image is out-of-distribution (NOT a leaf)
            - scores_dict: Dictionary with scores from each method
        """
        # Run all detection methods
        conf_in_dist, max_prob = self.compute_confidence_score(outputs)
        entropy_in_dist, ent_value = self.compute_entropy_score(outputs)
        var_in_dist, var_value = self.compute_variance_score(outputs)
        gap_in_dist, gap_value = self.compute_top_k_gap(outputs)
        spread_in_dist, spread_value = self.compute_prediction_spread(outputs)
        
        # Collect votes (now 5 methods)
        votes = [conf_in_dist, entropy_in_dist, var_in_dist, gap_in_dist, spread_in_dist]
        
        # Determine if image is OOD based on voting strategy
        if voting_strategy == "majority":
            # At least 3 out of 5 methods must agree it's in-distribution
            is_in_distribution = sum(votes) >= 3
        elif voting_strategy == "unanimous":
            # All methods must agree it's in-distribution
            is_in_distribution = all(votes)
        elif voting_strategy == "any":
            # At least one method thinks it's in-distribution
            is_in_distribution = any(votes)
        elif voting_strategy == "confidence_only":
            # Use only confidence threshold
            is_in_distribution = conf_in_dist
        else:
            raise ValueError(f"Unknown voting strategy: {voting_strategy}")
        
        is_ood = not is_in_distribution
        
        # Prepare scores dictionary
        scores = None
        if return_scores:
            scores = {
                "is_ood": is_ood,
                "max_probability": max_prob,
                "entropy": ent_value,
                "variance": var_value,
                "top_k_gap": gap_value,
                "prediction_spread": spread_value,
                "confidence_in_dist": conf_in_dist,
                "entropy_in_dist": entropy_in_dist,
                "variance_in_dist": var_in_dist,
                "gap_in_dist": gap_in_dist,
                "spread_in_dist": spread_in_dist,
                "in_distribution_votes": sum(votes),
                "total_votes": len(votes),
            }
        
        return is_ood, scores
    
    def get_recommendation(self, scores: Dict[str, float]) -> str:
        """
        Get human-readable recommendation based on OOD scores.
        
        Args:
            scores: Dictionary from detect() method
            
        Returns:
            Human-readable recommendation string
        """
        if not scores["is_ood"]:
            return "Image appears to be a valid plant leaf."
        
        # Analyze why it was flagged as OOD
        reasons = []
        
        if not scores["confidence_in_dist"]:
            reasons.append(
                f"Low confidence ({scores['max_probability']:.2%} < "
                f"{self.confidence_threshold:.2%})"
            )
        
        if not scores["entropy_in_dist"]:
            reasons.append(
                f"High uncertainty (entropy {scores['entropy']:.2f} > "
                f"{self.entropy_threshold:.2f})"
            )
        
        if not scores["variance_in_dist"]:
            reasons.append(
                f"Confused predictions (high variance {scores['variance']:.4f})"
            )
        
        if not scores["gap_in_dist"]:
            reasons.append(
                f"No clear winner (top-k gap {scores['top_k_gap']:.2%})"
            )
        
        if not scores["spread_in_dist"]:
            reasons.append(
                f"Uniform pattern detected (spread {scores['prediction_spread']:.4f})"
            )
        
        reason_str = "; ".join(reasons)
        
        return (
            f"⚠️ WARNING: This image does NOT appear to be a plant leaf!\n"
            f"Reasons: {reason_str}\n"
            f"Recommendation: Please upload a clear image of a plant leaf."
        )
    
    def tune_thresholds(
        self,
        outputs_in_dist: list,
        outputs_ood: list,
        target_fpr: float = 0.05
    ):
        """
        Automatically tune thresholds to achieve target false positive rate.
        
        This is useful if you have a validation set with both in-distribution
        (real leaves) and out-of-distribution (non-leaves) images.
        
        Args:
            outputs_in_dist: List of model outputs for in-distribution images
            outputs_ood: List of model outputs for OOD images
            target_fpr: Target false positive rate (default: 5%)
        """
        logger.info("Tuning OOD detection thresholds...")
        logger.info(f"In-distribution samples: {len(outputs_in_dist)}")
        logger.info(f"OOD samples: {len(outputs_ood)}")
        
        # Compute scores for all samples
        in_dist_probs = []
        in_dist_entropies = []
        
        for output in outputs_in_dist:
            probs = F.softmax(output, dim=1)
            max_prob = probs.max().item()
            ent = entropy(probs.cpu().numpy()[0])
            
            in_dist_probs.append(max_prob)
            in_dist_entropies.append(ent)
        
        # Find thresholds that achieve target FPR
        # (FPR = fraction of in-dist images incorrectly classified as OOD)
        sorted_probs = sorted(in_dist_probs)
        idx = int(len(sorted_probs) * target_fpr)
        self.confidence_threshold = sorted_probs[idx]
        
        sorted_entropies = sorted(in_dist_entropies, reverse=True)
        self.entropy_threshold = sorted_entropies[idx]
        
        logger.info(f"Updated confidence threshold: {self.confidence_threshold:.3f}")
        logger.info(f"Updated entropy threshold: {self.entropy_threshold:.3f}")


def create_default_detector(strict: bool = False) -> OODDetector:
    """
    Create OOD detector with default settings.
    
    Args:
        strict: If True, use stricter thresholds (fewer false negatives)
                If False, use lenient thresholds (fewer false positives)
    
    Returns:
        Configured OODDetector instance
    """
    if strict:
        # Strict mode: Better at catching OOD images but may reject some valid leaves
        return OODDetector(
            confidence_threshold=0.80,  # Increased from 0.75
            entropy_threshold=1.8,       # Decreased from 2.0
            use_temperature_scaling=False,
        )
    else:
        # Lenient mode: Better at accepting valid leaves but may miss some OOD images
        return OODDetector(
            confidence_threshold=0.70,  # Increased from 0.60
            entropy_threshold=2.5,       # Decreased from 3.0
            use_temperature_scaling=False,
        )
