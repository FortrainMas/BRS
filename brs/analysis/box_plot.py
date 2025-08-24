import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .analysis import DataInspectionStrategy

class BoxPlotInspection(DataInspectionStrategy):
    def __init__(self, column_name, title=None, figsize:tuple[int, int]=(8, 6)):
        super().__init__()
        self.column_name = column_name
        self.title = title
        self.figsize = figsize

    def inspect(self, df: pd.DataFrame):
        if self.column_name not in df.columns:
            raise ValueError(f"Column '{self.column_name}' not found in the dataframe.")

        plt.figure(figsize=self.figsize)

        sns.boxplot(y=df[self.column_name], color=sns.color_palette("Set2")[0])

        if self.title:
            plt.title(self.title)

        plt.show()
