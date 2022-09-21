from typing import List, Dict

from data.data_classes import DataInstance, FileData, Label


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


def _normalize_group_fairness_score(group_fairness: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Normalizes group fairness scores for sensitive attribute values to be positive if they contribute to fair
    outcomes, and negative if they contribute to unfair outcomes
    """

    normalized_fairness = {}

    # TODO: Determine a good way to perform this normalization. It is currently done by subtracting the mean from
    #  each value

    for value_name in group_fairness.keys():
        value_dict = group_fairness[value_name]
        mean_score = sum(value_dict.values()) / len(value_dict.values())

        normalized_values = {}
        for value in value_dict.keys():
            normalized_values[value] = value_dict[value] - mean_score

        normalized_fairness[value_name] = normalized_values

    return normalized_fairness


def set_labels(file_data: FileData, sensitive_attributes: List[str]) -> FileData:
    data_instances = file_data.instances
    group_fairness = {}

    for attribute in sensitive_attributes:
        group_fairness[attribute] = _compute_group_fairness(attribute, data_instances)
        print(group_fairness[attribute])



    normalized_fairness = _normalize_group_fairness_score(group_fairness)

    for instance in data_instances:
        fairness_score = 0
        for attribute in sensitive_attributes:
            instance_value = instance.inputs[attribute]
            fairness_score += normalized_fairness[attribute][instance_value]

        if fairness_score >= 0:
            instance.label = Label.FAIR
        else:
            instance.label = Label.UNFAIR

    return file_data
