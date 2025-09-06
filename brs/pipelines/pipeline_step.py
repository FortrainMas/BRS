from abc import ABC, abstractmethod

class PipelineStep(ABC):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def __call__(self, X, y=None):
        return self.fit_transform(X, y)
