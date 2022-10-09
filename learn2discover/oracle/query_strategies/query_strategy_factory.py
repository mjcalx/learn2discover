from loggers.logger_factory import LoggerFactory
from oracle.query_strategies.query_strategy import QueryStrategy
from oracle.query_strategies.least_confidence_strategy import LeastConfidenceStrategy
from oracle.query_strategies.entropy_strategy import EntropyStrategy

class QueryStrategyFactory:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.DEFAULT = LeastConfidenceStrategy

    def get_strategy(self, type_selection: str) -> QueryStrategy:
        strategy_types = {
            'entropy' : EntropyStrategy,
            'least_confidence' : LeastConfidenceStrategy,
        }

        try:
            strategy = strategy_types[type_selection]()
        except KeyError:
            self.logger.debug(f'QueryStrategy for "{type_selection}" not found. Defaulting to "least_confidence"')
            strategy = self.DEFAULT()
        finally:
            self.logger.info(f'Select QueryStrategy for "{type_selection}" : {str(strategy.name)}')        
        return strategy