from collections import defaultdict
from typing import List, Tuple, Union

import pandas as pd
import polars as pl
from replay.metrics import MRR, NDCG, Recall, OfflineMetrics


class ColdStartOfflineMetrics:
    # WARNING: `ColdStartOfflineMetrics` uses an averaging trick to compute overall metrics.
    # If you plan to add a new metric, ensure that this approach is appropriate for it.
    AVAILABLE_METRICS = {
        'MRR': MRR,
        'NDCG': NDCG,
        'Recall': Recall,
    }

    def __init__(
        self,
        metrics: Union[List[str], str] = ['MRR', 'NDCG', 'Recall'],
        topk: Union[List[int], int] = [5, 10, 20],
        user_column: str = 'user_id',
        item_column: str = 'item_id',
        rating_column: str = 'rating',
        cold_flag_column: str = 'is_cold',
    ) -> None:
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.cold_flag_column = cold_flag_column

        if isinstance(metrics, str):
            metrics = [metrics]

        if isinstance(topk, int):
            topk = [topk]

        self.metrics = []
        self.topk = topk

        for metric in metrics:
            if metric in self.AVAILABLE_METRICS:
                self.metrics.append(self.AVAILABLE_METRICS[metric](self.topk))
            else:
                msg = f'Metric "{metric}" is not available.'
                raise ValueError(msg)

        self.offline_metrics = OfflineMetrics(
            self.metrics,
            query_column=user_column,
            item_column=item_column,
            rating_column=rating_column,
        )

    def __call__(
        self,
        predictions: Union[pd.DataFrame, pl.DataFrame],
        ground_truth: Union[pd.DataFrame, pl.DataFrame],
    ) -> dict:
        if isinstance(predictions, pd.DataFrame) and isinstance(ground_truth, pd.DataFrame):
            self._check_cold_flag_column_pandas(ground_truth)
            condition = ground_truth[self.cold_flag_column]
            select = self._select_pandas
        elif isinstance(predictions, pl.DataFrame) and isinstance(ground_truth, pl.DataFrame):
            self._check_cold_flag_column_polars(ground_truth)
            condition = pl.col(self.cold_flag_column)
            select = self._select_polars
        else:
            msg = (
                'Expected `predictions` and `ground_truth` to be both either pandas.DataFrame or '
                f'polars.DataFrame. Found {type(predictions)} and {type(ground_truth)} instead.'
            )
            raise TypeError(msg)

        cold_predictions, cold_ground_truth = select(predictions, ground_truth, condition)
        num_cold = len(cold_ground_truth)
        cold_metrics = self._compute_subset_metrics(cold_predictions, cold_ground_truth)

        warm_predictions, warm_ground_truth = select(predictions, ground_truth, ~condition)
        num_warm = len(warm_ground_truth)
        warm_metrics = self._compute_subset_metrics(warm_predictions, warm_ground_truth)

        output = {}

        for metric in self.metrics:
            for k in self.topk:
                key = f'{metric.__name__}@{k}'

                output[f'cold_{key}'] = cold_metrics[key]
                output[f'warm_{key}'] = warm_metrics[key]

                output[key] = (cold_metrics[key] * num_cold + warm_metrics[key] * num_warm) / (
                    num_cold + num_warm
                )

        return output

    def _check_cold_flag_column_pandas(
        self, ground_truth: Union[pd.DataFrame, pl.DataFrame]
    ) -> None:
        if self.cold_flag_column not in ground_truth.columns:
            msg = f"Column '{self.cold_flag_column}' not found in `ground_truth`."
            raise RuntimeError(msg)

        if not pd.api.types.is_bool_dtype(ground_truth[self.cold_flag_column].dtype):
            msg = f"Column '{self.cold_flag_column}' must be of type bool."
            raise RuntimeError(msg)

    def _check_cold_flag_column_polars(
        self, ground_truth: Union[pd.DataFrame, pl.DataFrame]
    ) -> None:
        if self.cold_flag_column not in ground_truth.columns:
            msg = f"Column '{self.cold_flag_column}' not found in `ground_truth`."
            raise RuntimeError(msg)

        if ground_truth.get_column(self.cold_flag_column).dtype != pl.Boolean:
            msg = f"Column '{self.cold_flag_column}' must be of type bool."
            raise RuntimeError(msg)

    def _select_pandas(
        self, predictions: pd.DataFrame, ground_truth: pd.DataFrame, condition: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ground_truth = ground_truth[condition]
        return (
            predictions[predictions[self.user_column].isin(ground_truth[self.user_column])],
            ground_truth,
        )

    def _select_polars(
        self, predictions: pl.DataFrame, ground_truth: pl.DataFrame, condition: pl.Expr
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        ground_truth = ground_truth.filter(condition)
        return (
            predictions.filter(
                pl.col(self.user_column).is_in(ground_truth.get_column(self.user_column))
            ),
            ground_truth,
        )

    def _compute_subset_metrics(
        self,
        predictions: Union[pd.DataFrame, pl.DataFrame],
        ground_truth: Union[pd.DataFrame, pl.DataFrame],
    ) -> dict:
        if len(ground_truth) != 0:
            return self.offline_metrics(predictions, ground_truth)
        else:
            return defaultdict(lambda: 0.0)
