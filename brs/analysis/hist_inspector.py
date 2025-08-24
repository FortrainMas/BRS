import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .analysis import DataInspectionStrategy

class HistogramInspection(DataInspectionStrategy):
    def __init__(self, column_name, bins=10, title=None, discrete=False, figsize: tuple[int, int]=(8, 5)):
        super().__init__()
        self.column_name = column_name
        self.bins = bins
        self.title = title
        self.discrete = discrete
        self.figsize = figsize

    def inspect(self, df: pd.DataFrame):
        if self.column_name not in df.columns:
            raise ValueError(f"Column '{self.column_name}' not found in the dataframe.")

        data = df[self.column_name]

        plt.figure(figsize=self.figsize)
        sns.histplot(data, bins=self.bins, discrete=self.discrete, kde=False, color=sns.color_palette("Set2")[0])
        plt.xlabel(self.column_name)
        plt.ylabel("Frequency")
        if self.title:
            plt.title(self.title)
        plt.show()
