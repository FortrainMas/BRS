import pandas as pd
import numpy as np

from .metrics import MetricsStrategy

class HitRateAtK(MetricsStrategy):
    def __init__(self, k: int = 10):
        self.k = k

    def estimate(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> float:
        hits = 0
        users = y_true["user_id"].unique()

        for user in users:
            true_items = set(y_true[y_true["user_id"] == user]["item_id"].values)
            top_k_items = (
                y_pred[y_pred["user_id"] == user]
                .sort_values("score", ascending=False)
                .head(self.k)["item_id"]
                .values
            )
            if len(true_items.intersection(top_k_items)) > 0:
                hits += 1

        return hits / len(users)


class RecallAtK(MetricsStrategy):
    def __init__(self, k: int = 10):
        self.k = k

    def estimate(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> float:
        users = y_true["user_id"].unique()
        recall_sum = 0

        for user in users:
            true_items = set(y_true[y_true["user_id"] == user]["item_id"].values)
            top_k_items = (
                y_pred[y_pred["user_id"] == user]
                .sort_values("score", ascending=False)
                .head(self.k)["item_id"]
                .values
            )
            recall_sum += len(true_items.intersection(top_k_items)) / max(1, len(true_items))

        return recall_sum / len(users)

class PrecisionAtK(MetricsStrategy):
    def __init__(self, k: int = 10):
        self.k = k

    def estimate(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> float:
        users = y_true["user_id"].unique()
        precision_sum = 0

        for user in users:
            true_items = set(y_true[y_true["user_id"] == user]["item_id"].values)
            top_k_items = (
                y_pred[y_pred["user_id"] == user]
                .sort_values("score", ascending=False)
                .head(self.k)["item_id"]
                .values
            )
            precision_sum += len(true_items.intersection(top_k_items)) / max(1, self.k)

        return precision_sum / len(users)

class NDCGAtK(MetricsStrategy):
    def __init__(self, k: int = 10):
        self.k = k

    def estimate(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> float:
        users = y_true["user_id"].unique()
        ndcg_sum = 0

        for user in users:
            true_items = set(y_true[y_true["user_id"] == user]["item_id"].values)
            top_k_items = (
                y_pred[y_pred["user_id"] == user]
                .sort_values("score", ascending=False)
                .head(self.k)["item_id"]
                .values
            )
            dcg = 0
            idcg = 0
            for i, item in enumerate(top_k_items):
                if item in true_items:
                    dcg += 1 / np.log2(i + 2)
                idcg += 1 / np.log2(i + 2)
            ndcg_sum += dcg / max(1, idcg)

        return ndcg_sum / len(users)
