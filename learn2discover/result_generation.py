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

_FILL = 60
_RUN_MSG = lambda i : '#'*_FILL + " RUN {} ".format(i).ljust(9, '#') + '#'*_FILL

@app.command()
def generate_results(runs:int=100):
    """
    Generate summary results for a fixed number of runs of l2d.
    Recommend setting log level to warn in config.yml so that excessive logging is not generated.
    """
    # Initialise data structures
    df_accumulate = None
    dict_accumulate = None

    # Run l2d 'runs' times
    for i in progressbar(range(runs), redirect_stdout=True):
        print(_RUN_MSG(i+1))
        main() # run l2d

        # Get report path
        report_dir = os.path.join("report", Reporter.get_report_dir())

        latest_raw = pd.read_csv(os.path.join(report_dir, "raw_data.csv")) # load report
        with open(os.path.join(report_dir, "stats.json")) as f:
            latest_stats = json.load(f)

        if df_accumulate is None:
            df_accumulate = latest_raw
            dict_accumulate = latest_stats
        
        else:
            # Collect results from each run
            df_accumulate = df_accumulate.add(latest_raw, fill_value=0)
            for k in dict_accumulate.keys():
                dict_accumulate[k] += latest_stats[k]

    # Average the data
    df_mean = df_accumulate.div(runs)
    dict_mean = {k: v / runs for k, v in dict_accumulate.items()}

    print(df_mean)
    print(dict_mean)

    # Generate report
    output_csv_path = os.path.join(os.getcwd(), OUTPUT_CSV)
    df_mean.to_csv(output_csv_path, index=False)
    summary_plot(output_csv_path, os.path.join(os.getcwd(), OUTPUT_HTML))
    summary_dict(os.path.join(os.getcwd(), OUTPUT_STATS_JSON), dict_mean)

if __name__ == "__main__":
    app()