from typing import Union

import pandas as pd
import polars as pl


class _BaseFilter:
    def __call__(
        self, interactions: Union[pd.DataFrame, pl.DataFrame]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        if isinstance(interactions, pd.DataFrame):
            return self._filter_pandas(interactions)
        elif isinstance(interactions, pl.DataFrame):
            return self._filter_polars(interactions)
        else:
            msg = (
                'Expected input to be either pandas.DataFrame or polars.DataFrame. '
                f'Got {type(interactions)} instead.'
            )
            raise TypeError(msg)
