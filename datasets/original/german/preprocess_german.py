"""Utility for pre-processing german_credit data
"""
import csv
from typing import List

DURATION_IN_MONTH_COL = 1
CREDIT_AMOUNT_COL = 4
AGE_COL = 12


def months_to_years(row: List):
    """Converts the duration_in_months to a simpler number of years category.
    The number of years converted to is an integer number of full years passed.

    e.g. if the number of months is 6, then the number of years is 0

    Args:
        row (List): A single data instance
    """
    row[DURATION_IN_MONTH_COL] = int(row[DURATION_IN_MONTH_COL]) // 12


def credit_amount_quantile_upper_bound(row: List):
    """Converts the credit_amount to a more simple quantile in increments of 3000

    Args:
        row (List): A single data instance
    """

    range_quantile = int(row[CREDIT_AMOUNT_COL]) // 3000

    new_credit_range = f"<= {(range_quantile + 1) * 3000}"

    row[CREDIT_AMOUNT_COL] = new_credit_range


def age_to_range(row: List):
    """Converts the age to a simpler age_range category

    Args:
        row (List): A single data instance
    """
    age = int(row[AGE_COL]) // 30

    switcher = {
        0: "<= 30",
        1: "<= 60",
        2: "> 60"
    }

    row[AGE_COL] = (switcher.get(age))


def run():
    """Run the preprocessor
    """
    rows = []

    with open('german.csv', 'r') as readFile:
        reader = csv.reader(readFile)

        # Ignore the first column as it is just an index
        headers = next(reader)[1:]

        headers[DURATION_IN_MONTH_COL] = "duration_in_full_years"
        headers[CREDIT_AMOUNT_COL] = "credit_amount_range"
        headers[AGE_COL] = "age_range"

        rows.append(headers)
        for row in reader:
            row = row[1:]

            # Perform numeric to categorical conversions
            months_to_years(row)
            credit_amount_quantile_upper_bound(row)
            age_to_range(row)
            rows.append(row)

    with open('german-preprocessed.csv', 'w') as writeFile:
        writer = csv.writer(writeFile, delimiter=",", lineterminator="\n")
        writer.writerows(rows)


if __name__ == "__main__":
    run()
