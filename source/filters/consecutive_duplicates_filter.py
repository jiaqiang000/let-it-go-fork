from typing import Literal
from uuid import uuid4

import pandas as pd
import polars as pl

from ._filter import _BaseFilter


class ConsecutiveDuplicatesFilter(_BaseFilter):
    def __init__(
        self,
        keep: Literal['first', 'last'] = 'first',
        user_column: str = 'user_id',
        item_column: str = 'item_id',
        timestamp_column: str = 'timestamp',
    ) -> None:
        super().__init__()
        self.user_column = user_column
        self.item_column = item_column
        self.timestamp_column = timestamp_column

        if keep not in ('first', 'last'):
            msg = 'Argument `keep` must be either "first" or "last".'
            raise ValueError(msg)

        self.bias = 1 if keep == 'first' else -1
        self.temporary_column = f'__shifted_{uuid4().hex[:8]}'

    def _filter_pandas(self, interactions: pd.DataFrame) -> pd.DataFrame:
        interactions = interactions.sort_values([self.user_column, self.timestamp_column])
        interactions[self.temporary_column] = interactions.groupby(self.user_column)[
            self.item_column
        ].shift(periods=self.bias)
        return (
            interactions[interactions[self.item_column] != interactions[self.temporary_column]]
            .drop(self.temporary_column, axis=1)
            .reset_index(drop=True)
        )

    def _filter_polars(self, interactions: pl.DataFrame) -> pl.DataFrame:
        return (
            interactions.sort(self.user_column, self.timestamp_column)
            .with_columns(
                pl.col(self.item_column)
                .shift(n=self.bias)
                .over(self.user_column)
                .alias(self.temporary_column)
            )
            .filter((pl.col(self.item_column) != pl.col(self.temporary_column)).fill_null(True))
            .drop(self.temporary_column)
        )
