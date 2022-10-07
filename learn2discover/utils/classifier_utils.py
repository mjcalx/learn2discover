import math
import torch
from typing import List

class ClassifierUtils:

    @staticmethod
    def get_confidence_from_log_probs(log_probs: List[torch.Tensor]):
        log_prob = log_probs.data.tolist()[0][1]
        prob = math.exp(log_prob)
        confidence = prob if prob >= 0.5 else 1 - prob
        return confidence