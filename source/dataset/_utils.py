import os

import polars as pl


def load_data(filepath: str) -> pl.DataFrame:
    _, ext = os.path.splitext(filepath)

    match ext:
        case '.csv':
            return pl.read_csv(filepath)
        case '.parquet':
            return pl.read_parquet(filepath)
        case _:
            msg = f'File format {ext} is not supported.'
            raise RuntimeError(msg)
