import numpy as np
import pandas as pd
import plotly

from json import dump
from plotly import subplots

from typing import Dict

def summary_plot(csv_filepath: str, output_file: str, rows: int = 25, yLim: int = 1):
        """
        Adapted from https://hatefdastour.github.io/portfolio/financial_analysis_and_modeling/Credit_Card_Fraud_Detection_PyTorch_MLP.html
        """

        data = pd.read_csv(csv_filepath)
        
        fig = subplots.make_subplots(rows=1, cols=2, horizontal_spacing = 0.02, column_widths=[0.6, 0.4],
                            specs=[[{"type": "scatter"},{"type": "table"}]])
        # Left
        fig.add_trace(plotly.graph_objs.Scatter(x= data['Annotations'].values, y= data['Precision'].astype(float).values.round(4),
                                line=dict(color='OrangeRed', width= 1.5), name = 'Precision'), 1, 1)
        fig.add_trace(plotly.graph_objs.Scatter(x= data['Annotations'].values, y= data['Accuracy'].astype(float).values,
                                line=dict(color='MidnightBlue', width= 1.5),  name = 'Accuracy'), 1, 1)
        fig.add_trace(plotly.graph_objs.Scatter(x= data['Annotations'].values, y= data['Confidence'].astype(float).values,
                                line=dict(color='ForestGreen', width= 1.5),  name = 'Confidence'), 1, 1)
        fig.update_layout(legend=dict(x=0, y=1.1, traceorder='reversed', font_size=12),
                    dragmode='select', plot_bgcolor= 'white', height=600, hovermode='closest',
                    legend_orientation='h')
        fig.update_xaxes(range=[data.Annotations.min(), data.Annotations.max()],
                        showgrid=True, gridwidth=1, gridcolor='Lightgray',
                        showline=True, linewidth=1, linecolor='Lightgray', mirror=True, row=1, col=1)
        fig.update_yaxes(range=[0, yLim], showgrid=True, gridwidth=1, gridcolor='Lightgray',
                        showline=True, linewidth=1, linecolor='Lightgray', mirror=True, row=1, col=1)
        # Right
        ind = np.linspace(0, data.shape[0], rows, endpoint = False).round(0).astype(int)
        ind = np.append(ind, data.index[-1])
        h = data[data.index.isin(ind)]
        T = h.copy()
        T[['Loss','Accuracy']] = T[['Loss','Accuracy']].applymap(lambda x: '%.4e' % x)
        Temp = []
        for i in T.columns:
            Temp.append(T.loc[:,i].values)
        fig.add_trace(plotly.graph_objs.Table(header=dict(values = list(data.columns), line_color='darkslategray',
                                        fill_color='Navy', align=['center','center'],
                                        font=dict(color='white', size=12), height=25), columnwidth = [0.4, 0.4, 0.4],
                            cells=dict(values=Temp, line_color='darkslategray',
                                        fill=dict(color=['Lavender', 'white', 'white']),
                                        align=['center', 'center'], font_size=12,height=20)), 1, 2)
        fig.write_html(output_file)

def summary_dict(output_path: str, result: Dict[str, object]):
    with open(output_path, "w") as f:
        dump(result, f, indent=4)
