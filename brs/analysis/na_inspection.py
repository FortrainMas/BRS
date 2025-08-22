import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .analysis import DataInspectionStrategy

class NAInspection(DataInspectionStrategy):
    def __init__(self, dataset_name: str = "", percent_precision: int = 2):

        """
        Constructor for the BarChartInspection class.

        Parameters:
        dataset_name (str): The name of the dataset to be analyzed (it is only to be printed in report). Defaults to ""
        percent_precision (int): The number of decimal places to round the percentage to. Defaults to 2.
        """

        super().__init__()
        self.dataset_name = dataset_name
        self.percent_precision = percent_precision

    def inspect(self, df: pd.DataFrame):
        
        """
        Print the total size of the given DataFrame and the count of NaN values
        in each column as an absolute value and as a percentage of the total size.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be inspected.
        """

        print(f"{self.dataset_name} total size {len(df)}")
        total = len(df)
        na_stats = df.isna().sum().to_frame("NaN count")
        na_stats["NaN %"] = (na_stats["NaN count"] / total * 100).round(self.percent_precision).astype(str) + "%"
        print(na_stats)
