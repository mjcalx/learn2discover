from oracle.query_strategies.query_strategy import QueryStrategy

class LeastConfidenceStrategy(QueryStrategy):
    def __name__(self):
        return 'Least Confidence Strategy'