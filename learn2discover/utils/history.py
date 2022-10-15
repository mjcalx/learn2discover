'''
A data type to keep track of the history of the oracle, using a pandas data frame for column-based data.
'''
import pandas as pd
import numpy as np

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
    
    @staticmethod
    def reset():
        History.instance = None
        
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
