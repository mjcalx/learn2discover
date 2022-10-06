import pandas as pd
import torch.nn as nn
from oracle.query_strategies.query_strategy import QueryStrategy

class MarginSamplingStrategy(QueryStrategy):
    @property
    def name(self):
        return 'Margin Sampling Strategy'

    def query(self, classifier: nn.Module) -> pd.DataFrame:
        raise NotImplementedError