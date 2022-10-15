import json
import os
import pandas as pd
import typer
from progressbar import progressbar

from l2d import main
from utils.summary import summary_plot, summary_dict
from utils.reporter import Reporter

app = typer.Typer()

OUTPUT_CSV = "report/final_raw_data.csv"
OUTPUT_STATS_JSON = "report/final_stats.json"
OUTPUT_HTML = "report/final_report.html"

_RUN_MSG = '#'*60 + " RUN {} " + '#'*60

@app.command()
def generate_results(runs:int=100):
    """
    Generate summary results for a fixed number of runs of l2d.
    Recommend setting log level to warn in config.yml so that excessive logging is not generated.
    """
    # Initialise data structures
    raw_data_df = None
    result_dict = None

    # Run l2d 'runs' times
    for i in progressbar(range(runs), redirect_stdout=True):
        print(_RUN_MSG.format(i+1))
        main() # run l2d

        # Get report path
        report_dir = os.path.join("report", Reporter.get_report_dir())

        latest_raw = pd.read_csv(os.path.join(report_dir, "raw_data.csv")) # load report
        with open(os.path.join(report_dir, "stats.json")) as f:
            latest_stats = json.load(f)

        if raw_data_df is None:
            raw_data_df = latest_raw
            result_dict = {k:[] for k in latest_stats.keys()}
        
        else:
            # Collect results from each run
            raw_data_df = pd.concat([raw_data_df, latest_raw])
            for k in result_dict.keys():
                result_dict[k].append(latest_stats[k])

    # Average the data
    mean_df   = raw_data_df.groupby(lambda idx: idx).mean()
    mean_dict = {k:sum(v)/len(v) for k,v in result_dict.items()}
        
    # Generate report
    output_csv_path = os.path.join(os.getcwd(), OUTPUT_CSV)
    mean_df.to_csv(output_csv_path, index=False)
    summary_plot(output_csv_path, os.path.join(os.getcwd(), OUTPUT_HTML))
    summary_dict(os.path.join(os.getcwd(), OUTPUT_STATS_JSON), mean_dict)

if __name__ == "__main__":
    app()