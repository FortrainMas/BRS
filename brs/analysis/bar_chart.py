import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .analysis import DataInspectionStrategy

class BarChartInspection(DataInspectionStrategy):
    def __init__(self, column_name, top_n=8, title=None, dropna=False):
        super().__init__()
        self.column_name = column_name
        self.top_n = top_n
        self.title = title
        self.dropna = dropna

    def inspect(self, df: pd.DataFrame):
        if self.column_name not in df.columns:
            raise ValueError(f"Column '{self.column_name}' not found in the dataframe.")

        counts = df[self.column_name].value_counts(dropna=self.dropna)

        if len(counts) < self.top_n:
            self.top_n = len(counts)

        top_counts = counts.iloc[:self.top_n]

        plot_df = top_counts.reset_index()
        plot_df.columns = [self.column_name, 'count']

        plt.figure(figsize=(8, 0.5 * len(plot_df)))
        sns.barplot(
            x='count',
            y=self.column_name,
            data=plot_df,
            hue=self.column_name,
            palette='Set2',
            dodge=False,
        )
        if self.title:
            plt.title(self.title)
        plt.tight_layout()
        plt.show()
