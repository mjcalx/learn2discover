"""
Utility for pre-processing COMPAS data
"""

import csv

SCORE_TEXT_ROW = 24

def preprocess_compas():
    lines = list()
    with open('compas-scores-raw.csv', 'r') as readFile:
        reader = csv.reader(readFile)

        # Add header to new file
        header = next(reader)
        lines.append(header)

        for row in reader:
            # Remove invalid rows
            if row[SCORE_TEXT_ROW] in ["High", "Medium", "Low"]: 
                lines.append(row)
        
    with open('compas-scores-preprocessed.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)

if __name__ == "__main__":
    preprocess_compas()
