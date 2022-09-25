import pandas as pd
from typing import List, Dict
from data.data_classes import DataInstance, Label, Outcome, FileData
from oracle.dataset_manager import DatasetManager

def _compute_group_fairness(sensitive_attribute: str, instances: List[DataInstance]):
    """
    Computes the group fairness for a single attribute
    """

    # Store each of the attribute's values and corresponding group fairness in a dictionary
    group_fairness = {}

    # Add each unique value for the sensitive attribute to the dictionary
    for instance in instances:
        if instance.inputs[sensitive_attribute] not in group_fairness:
            group_fairness[instance.inputs[sensitive_attribute]] = 0

    # Compute the group fairness for each attribute
    for value in group_fairness.keys():
        pass_count = 0
        value_count = 0

        for instance in instances:
            if instance.inputs[sensitive_attribute] == value:
                value_count += 1
                if instance.outcome.value:
                    pass_count += 1

        group_fairness[value] = pass_count / value_count

    return group_fairness

def _compute_group_fairness(sensitive_attribute: str, possible_values: List[str], dmgr: DatasetManager):
    """
    Computes the group fairness for a single attribute
    """
    data_inputs = dmgr.X
    outcomes = dmgr.outcomes
    # Store each of the attribute's values and corresponding group fairness in a dictionary
    scores = {category:0 for category in possible_values}

    # Calculate score
    for sensitive_value in scores.keys():
        filtered = data_inputs[sensitive_attribute][lambda x : x == sensitive_value]
        value_count = len(filtered)
        pass_count = len(outcomes[filtered.index][ lambda x : x == Outcome.PASS ])
        scores[sensitive_value] = pass_count / value_count
    return scores

def _normalize_group_fairness_score(group_fairness_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]: # -> Dict[str,Series]
    """
    Normalizes group fairness scores for sensitive attribute values to be positive if they contribute to fair
    outcomes, and negative if they contribute to unfair outcomes
    """
    group_fairness = {attr:pd.Series(group_fairness_dict[attr]) for attr in group_fairness_dict.keys()}
    means = {key:group_fairness[key].mean() for key in group_fairness.keys()}
    # TODO: Determine a good way to perform this normalization. It is currently done by subtracting the mean from
    normalized_fairness = {key:group_fairness[key].apply(lambda x: x - means[key])  for key in group_fairness.keys()}
    return normalized_fairness

def set_labels(dmgr: DatasetManager, sensitive_attributes: List[str]) -> pd.Series:
    group_fairness = {}
    for attribute in sensitive_attributes:
        group_fairness[attribute] = _compute_group_fairness(attribute, dmgr.schema.get_variable_values(attribute), dmgr)
        print(group_fairness[attribute])

    normalized = calculate_fairness_scores(group_fairness, dmgr, sensitive_attributes)
    assign_fairness_label = lambda x : Label.FAIR if x >=0 else Label.UNFAIR
    return normalized.apply(assign_fairness_label)

def calculate_fairness_scores(group_fairness: Dict[str, float], dmgr: DatasetManager, sensitive_attributes):
    normalized_fairness = _normalize_group_fairness_score(group_fairness)
    sum_over_normalized = lambda x: sum([normalized_fairness[attr][x[attr]] for attr in sensitive_attributes])
    normalized_fairness_scores = dmgr.X.apply(sum_over_normalized, axis=1)
    return normalized_fairness_scores