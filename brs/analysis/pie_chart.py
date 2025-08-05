import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .analysis import DataInspectionStrategy

class PieChartInspection(DataInspectionStrategy):
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
        others_count = counts.iloc[self.top_n:].sum()
        
        if others_count > 0:
            top_counts = pd.concat([top_counts, pd.Series({"others": others_count})])
        plt.figure(figsize=(6,6))
        plt.pie(
            top_counts.values,
            labels=top_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2")[0:len(top_counts)]
        )
        if self.title:
            plt.title(self.title)
        plt.show()
