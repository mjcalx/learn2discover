import os
import sys
import pandas as pd
from typing import List, Dict

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
datagen_path = os.path.join(root_path, 'data_generation')
try:
    sys.path.index(datagen_path)
except ValueError:
    sys.path.append(datagen_path)

from data.data_classes import Label, Outcome
from data.dataset_manager import DatasetManager
from mock_oracle import MockOracle

class CompasOracle(MockOracle):
    def __init__(self, sensitive_attributes: List[str]):
        super(CompasOracle, self).__init__()
        self.sensitive_attributes = sensitive_attributes

    def set_labels(self) -> pd.Series:
        dataset_manager = DatasetManager.get_instance()
        self.logger.debug("Evaluating fairness...")
        group_fairness = {}
        for attribute in self.sensitive_attributes:
            group_fairness[attribute] = self._compute_group_fairness(attribute, dataset_manager.schema.get_variable_values(attribute))
            self.logger.debug(str(group_fairness[attribute]))

        normalized = self._calculate_fairness_scores(group_fairness)
        assign_fairness_label = lambda x : Label.FAIR if x >=0 else Label.UNFAIR
        fairness_labels = normalized.apply(assign_fairness_label)

        label_count = lambda label : len(fairness_labels[lambda x : x == label])
        fair_count   = label_count(Label.FAIR)
        unfair_count = label_count(Label.UNFAIR)
        self.logger.debug(f"Fair instances: {fair_count}. Unfair instances: {unfair_count}")
        self.logger.debug(f"Fair instances %: {fair_count / len(dataset_manager.data)}%. Unfair instances %: {unfair_count / len(dataset_manager.data)}%")

        return fairness_labels

    def _compute_group_fairness(self, sensitive_attribute: str, possible_values: List[str]):
        """
        Computes the group fairness for a single attribute
        """
        dataset_manager = DatasetManager.get_instance()
        data_inputs = dataset_manager.X
        outcomes = dataset_manager.outcomes
        # Store each of the attribute's values and corresponding group fairness in a dictionary
        scores = {category:0 for category in possible_values}

        # Calculate score
        for sensitive_value in scores.keys():
            filtered = data_inputs[sensitive_attribute][lambda x : x == sensitive_value]
            value_count = len(filtered)
            pass_count = len(outcomes[filtered.index][ lambda x : x == Outcome.PASS ])
            scores[sensitive_value] = pass_count / value_count
        return scores

    def _normalize_group_fairness_score(self, group_fairness_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Normalizes group fairness scores for sensitive attribute values to be positive if they contribute to fair
        outcomes, and negative if they contribute to unfair outcomes
        """
        group_fairness = {attr:pd.Series(group_fairness_dict[attr]) for attr in group_fairness_dict.keys()}
        means = {key:group_fairness[key].mean() for key in group_fairness.keys()}
        # TODO: Determine a good way to perform this normalization. It is currently done by subtracting the mean from
        normalized_fairness = {key:group_fairness[key].apply(lambda x: x - means[key])  for key in group_fairness.keys()}
        return normalized_fairness

    def _calculate_fairness_scores(self, group_fairness: Dict[str, float]):
        dataset_manager = DatasetManager.get_instance()
        normalized_fairness = self._normalize_group_fairness_score(group_fairness)
        sum_over_normalized = lambda x: sum([normalized_fairness[attr][x[attr]] for attr in self.sensitive_attributes])
        normalized_fairness_scores = dataset_manager.X.apply(sum_over_normalized, axis=1)
        return normalized_fairness_scores