import math
import torch
from typing import List

class ClassifierUtils:

    @staticmethod
    def get_confidence_from_log_probs(log_probs: List[torch.Tensor]) -> List[float]:
        # Get the log probs for a single label (doesn't matter which)
        single_label_log_probs = list(zip(*log_probs.data.tolist()))[0]
        probs = (math.exp(lp) for lp in single_label_log_probs)
        confidences = [p if p >= 0.5 else 1 - p for p in probs]
        return confidences