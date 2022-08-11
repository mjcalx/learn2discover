import numpy as np

class DatasetManager:
    def __init__(self, preprocessors=None, random_state=42):
        self.random = np.random.RandomState(random_state)
        self.preprocessors = preprocessors