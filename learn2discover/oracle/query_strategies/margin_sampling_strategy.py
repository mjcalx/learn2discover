from oracle.query_strategies.query_strategy import QueryStrategy

class MarginSamplingStrategy(QueryStrategy):
    def __name__(self):
        return 'Margin Sampling Strategy'