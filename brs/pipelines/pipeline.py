import pandas as pd

from brs.pipelines.pipeline_step import PipelineStep

class Pipeline(PipelineStep):
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps
    
    def fit(self, X, y=None) -> PipelineStep:
        for step in self.steps:
            X = step.fit(X)
        return self
    
    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X
