import json
import os
import pandas as pd
import typer

from l2d import main
from utils.summary import summary_plot, summary_dict
from utils.reporter import Reporter

app = typer.Typer()

OUTPUT_CSV = "report/final_raw_data.csv"
OUTPUT_STATS_JSON = "report/final_stats.json"
OUTPUT_HTML = "report/final_report.html"

@app.command()
def generate_results(runs:int=100):
    """
    Generate summary results for a fixed number of runs of l2d.
    Recommend setting log level to warn in config.yml so that excessive logging is not generated.
    """
    # Initialise data structures
    raw_data_df = None
    result_dict = {}

    # Run l2d 'runs' times
    for i in range(runs):
        print(f"Running l2d {i}")
        main() # run l2d

        # Get report path
        report_dir = os.path.join("report", Reporter.get_report_dir())

        latest_raw = pd.read_csv(os.path.join(report_dir, "raw_data.csv")) # load report
        with open(os.path.join(report_dir, "stats.json")) as f:
            latest_stats = json.load(f)

        if raw_data_df is None:
            raw_data_df = latest_raw
            result_dict = latest_stats
        
        else:
            # Sum the two data frames
            raw_data_df = raw_data_df.add(latest_raw, fill_value=0)

            # Sum the two dicts
            for k, v1 in result_dict.items():
                for k, v2 in latest_stats.items():
                    result_dict[k] = (v1 + v2)

        print(raw_data_df)
        print(result_dict)
    
    # Divide final values by 'runs' to get averages
    raw_data_df = raw_data_df.div(runs)
    result_dict = {k: v / runs for k, v in result_dict.items()}

    # Generate report
    output_csv_path = os.path.join(os.getcwd(), OUTPUT_CSV)
    raw_data_df.to_csv(output_csv_path, index=False)
    summary_plot(output_csv_path, os.path.join(os.getcwd(), OUTPUT_HTML))
    summary_dict(os.path.join(os.getcwd(), OUTPUT_STATS_JSON), result_dict)

if __name__ == "__main__":
    app()