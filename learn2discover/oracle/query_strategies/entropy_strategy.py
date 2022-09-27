import pandas as pd
import torch.nn as nn
from oracle.query_strategies.query_strategy import QueryStrategy

class EntropyStrategy(QueryStrategy):
    def name(self):
        return 'EntropyStrategy'

    def query(self, classifier: nn.Module) -> pd.DataFrame:
        raise NotImplementedError