from read_data import parse_data

INPUT_ATTRIBUTES = ['Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text', 'LastName', 'FirstName', 'MiddleName',
                    'Sex_Code_Text', 'Ethnic_Code_Text', 'DateOfBirth', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason',
                    'Language', 'LegalStatus', 'CustodyStatus', 'MaritalStatus', 'Screening_Date',
                    'RecSupervisionLevel', 'RecSupervisionLevelText', 'Scale_ID']
OUTPUT_ATTRIBUTES = ['DisplayText', 'RawScore', 'DecileScore', 'ScoreText', 'AssessmentType', 'IsCompleted',
                     'IsDeleted']


def read_compas_data(filepath: str):
    """
    Reads a COMPAS csv data file
    """
    return parse_data(filepath, INPUT_ATTRIBUTES, OUTPUT_ATTRIBUTES)


if __name__ == "__main__":
    testpath = "../../datasets/original/COMPAS/compas-scores-raw.csv"
    data = read_compas_data(testpath)
    print(data.instances[0].label)
