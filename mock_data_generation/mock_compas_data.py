from data_classes import DataInstance, TestOutcome
from create_mocked_data import get_mock_data, create_l2d_input_csv

INPUT_ATTRIBUTES = ['Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text', 'LastName', 'FirstName', 'MiddleName',
                    'Sex_Code_Text', 'Ethnic_Code_Text', 'DateOfBirth', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason',
                    'Language', 'LegalStatus', 'CustodyStatus', 'MaritalStatus', 'Screening_Date',
                    'RecSupervisionLevel', 'RecSupervisionLevelText', 'Scale_ID']
OUTPUT_ATTRIBUTES = ['DisplayText', 'RawScore', 'DecileScore', 'ScoreText', 'AssessmentType', 'IsCompleted',
                     'IsDeleted']

DATA_PATH = "../datasets/original/COMPAS/compas-scores-raw.csv"


def determine_compas_outcome(instance: DataInstance) -> TestOutcome:
    """
    Determines and returns the outcome for a given data instance in the COMPAS dataset
    """
    if instance.outputs["ScoreText"] in ["High", "Medium"]:
        return TestOutcome.FAIL
    else:
        return TestOutcome.PASS


if __name__ == "__main__":
    """
    A simple test example to show that everything is working
    """

    # Not intrinsic to the dataset, should be redefined arbitrarily for each run of the active learning process
    sample_sensitive_attributes = ["Sex_Code_Text", "Ethnic_Code_Text"]

    # Get the labelled data
    # data = get_compas_data(testpath, sample_sensitive_attributes)
    data = get_mock_data(DATA_PATH, INPUT_ATTRIBUTES, OUTPUT_ATTRIBUTES, sample_sensitive_attributes,
                         determine_compas_outcome)

    # print(DATA_PATH.split("/")[-1][:-4])
    create_l2d_input_csv(data)

    # # See how many instances are fair vs unfair
    # fair_count = 0
    # unfair_count = 0
    # pass_unfair_count = 0
    # fail_fair_count = 0
    # for instance in data.instances:
    #     if instance.label.value:
    #         fair_count += 1
    #     else:
    #         unfair_count += 1
    #
    #     if instance.outcome.value and not instance.label.value:
    #         pass_unfair_count += 1
    #
    #     if not instance.outcome.value and instance.label.value:
    #         fail_fair_count += 1
    #
    # print(f"Fair instances: {fair_count}. Unfair instances: {unfair_count}")
    # print(
    #     f"Fair instances %: {fair_count / len(data.instances)}%. Unfair instances %: {unfair_count / len(data.instances)}%")
    # print(f"Pass unfairly count: {pass_unfair_count}. Percent: {pass_unfair_count / len(data.instances)}")
    # print(f"Fail fairly count: {fail_fair_count}. Percent: {fail_fair_count / len(data.instances)}")
