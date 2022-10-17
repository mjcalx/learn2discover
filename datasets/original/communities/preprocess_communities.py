"""
Utility for preprocessing Communities and Crime data
"""
import csv
from typing import List

headers = ['state', 'county', 'community', 'communityname', 'fold', 'population', 'householdsize', 'racepctblack', 'racePctWhite', 
    'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 
    'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 
    'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov', 
    'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu', 
    'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 
    'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 
    'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 
    'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup', 
    'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 
    'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 
    'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ', 
    'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 
    'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 
    'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 
    'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 
    'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg', 'LemasPctOfficDrugUn', 
    'LemasPctPolicOnPatr', 'ViolentCrimesPerPop', 'LemasGangUnitDeploy', 'PolicBudgPerPop']

input_csv = "communities-raw.csv"
output_csv = "communities-preprocessed.csv"

REMOVE_END = 27
KEEP_END = 3
RACEPCT_COLS = [6,7,8,9]
MEDINCOME_COL = 16

def racepct_to_category(row:List):
    """Converts a racepct value to a string category

    Args:
        row (List): A single data instance
    """
    for col in RACEPCT_COLS:
        i = int(float(row[col])*5)

        switcher = {
            0: "<=0.2",
            1: "<=0.4",
            2: "<=0.6",
            3: "<=0.8",
            4: "<1",
            5: "1"
        }

        row[col] = (switcher.get(i))

    return row

def medincome_to_category(row:List):
    """Split median income into categories

    Args:
        row (List): A single data instance
    """
    income_category = int(float(row[MEDINCOME_COL])*3)

    switcher = {
        0: "Low",
        1: "Medium",
        2: "High",
        3: "High",
    }

    row[MEDINCOME_COL] = (switcher.get(income_category))

    return row


def process_row(row:List):
    """Fully preprocess a row

    Args:
        row (List): A single data instance
    """
    return medincome_to_category(racepct_to_category(row))

def preprocess_communities():
    lines = list()
    with open(input_csv, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            # Remove columns with bad data, keep outcome variable
            del row[3]
            del row[-REMOVE_END:-KEEP_END]
            del row[-KEEP_END+1:]
            # Only include rows with no '?' (unknown) values
            if '?' not in row: 
                lines.append(process_row(row))
        
    with open(output_csv, 'w') as writeFile:
        writer = csv.writer(writeFile)
        # Write headers
        del headers[3]
        del headers[-REMOVE_END:-KEEP_END]
        del headers[-KEEP_END+1:]
        writer.writerow(headers)
        # Write rows
        writer.writerows(lines)

if __name__ == "__main__":
    preprocess_communities()