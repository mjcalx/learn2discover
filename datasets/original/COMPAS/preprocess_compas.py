"""
Utility for pre-processing COMPAS data
"""

import csv

SCORE_TEXT_ROW = 24

lines = list()
with open('compas-scores-raw.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    for row in reader:
        # Remove invalid rows
        if row[24] in ["High", "Medium", "Low"]: 
            lines.append(row)
with open('compas-scores-preprocessed.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)

