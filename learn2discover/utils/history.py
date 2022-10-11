'''
A data type to keep track of the history of the oracle, using a pandas data frame for column-based data.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
from plotly import subplots

df_cols =['Iteration', 'Loss', 'Annotations', 'Accuracy', 'Precision', 'Confidence']

class History():
    instance = None

    def __init__(self):
        self.data = pd.DataFrame(
            [np.zeros(6)],
            columns=df_cols
        )
    
    @staticmethod
    def get_instance():
        if History.instance is None:
            History.instance = History()
        return History.instance
        
    def concat(self, iter, loss, annotations, accuracy, precision, confidence):
        self.data = pd.concat(
            [self.data, 
            pd.DataFrame(
                [[iter, 
                loss, 
                annotations, 
                accuracy, 
                precision,
                confidence]],
                columns = df_cols)]
        )

    def plot_history(self, filename, rows = 25, yLim = 1):
        """
        Adapted from https://hatefdastour.github.io/portfolio/financial_analysis_and_modeling/Credit_Card_Fraud_Detection_PyTorch_MLP.html
        """
        
        fig = subplots.make_subplots(rows=1, cols=2, horizontal_spacing = 0.02, column_widths=[0.6, 0.4],
                            specs=[[{"type": "scatter"},{"type": "table"}]])
        # Left
        fig.add_trace(plotly.graph_objs.Scatter(x= self.data['Annotations'].values, y= self.data['Precision'].astype(float).values.round(4),
                                line=dict(color='OrangeRed', width= 1.5), name = 'Precision'), 1, 1)
        fig.add_trace(plotly.graph_objs.Scatter(x= self.data['Annotations'].values, y= self.data['Accuracy'].astype(float).values,
                                line=dict(color='MidnightBlue', width= 1.5),  name = 'Accuracy'), 1, 1)
        fig.add_trace(plotly.graph_objs.Scatter(x= self.data['Annotations'].values, y= self.data['Confidence'].astype(float).values,
                                line=dict(color='ForestGreen', width= 1.5),  name = 'Confidence'), 1, 1)
        fig.update_layout(legend=dict(x=0, y=1.1, traceorder='reversed', font_size=12),
                    dragmode='select', plot_bgcolor= 'white', height=600, hovermode='closest',
                    legend_orientation='h')
        fig.update_xaxes(range=[self.data.Annotations.min(), self.data.Annotations.max()],
                        showgrid=True, gridwidth=1, gridcolor='Lightgray',
                        showline=True, linewidth=1, linecolor='Lightgray', mirror=True, row=1, col=1)
        fig.update_yaxes(range=[0, yLim], showgrid=True, gridwidth=1, gridcolor='Lightgray',
                        showline=True, linewidth=1, linecolor='Lightgray', mirror=True, row=1, col=1)
        # Right
        ind = np.linspace(0, self.data.shape[0], rows, endpoint = False).round(0).astype(int)
        ind = np.append(ind, self.data.index[-1])
        h = self.data[self.data.index.isin(ind)]
        T = h.copy()
        T[['Loss','Accuracy']] = T[['Loss','Accuracy']].applymap(lambda x: '%.4e' % x)
        Temp = []
        for i in T.columns:
            Temp.append(T.loc[:,i].values)
        fig.add_trace(plotly.graph_objs.Table(header=dict(values = list(self.data.columns), line_color='darkslategray',
                                        fill_color='Navy', align=['center','center'],
                                        font=dict(color='white', size=12), height=25), columnwidth = [0.4, 0.4, 0.4],
                            cells=dict(values=Temp, line_color='darkslategray',
                                        fill=dict(color=['Lavender', 'white', 'white']),
                                        align=['center', 'center'], font_size=12,height=20)), 1, 2)
        fig.show()
        fig.write_html(filename)
    