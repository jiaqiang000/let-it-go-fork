import polars as pl

from ._dataset import _SequentialDataset, _TestMixin, _TrainMixin


class _CausalMixin:
    def __init__(self, **kwargs) -> None:
        super(_CausalMixin, self).__init__(**kwargs)

        if self.add_labels:
            self.max_length += 1

    def __getitem__(self, index: int) -> dict:
        item = super(_CausalMixin, self).__getitem__(index)

        if self.add_labels:
            item['labels'] = item['inputs'][1:].clone()
            item['inputs'] = item['inputs'][:-1]

        return item


class TestCausalDataset(_TestMixin, _CausalMixin, _SequentialDataset):
    def __init__(
        self,
        interactions: pl.DataFrame,
        ground_truth: pl.DataFrame | None = None,
        add_labels: bool = False,
        user_column: str = 'user_id',
        item_column: str = 'item_id',
        timestamp_column: str = 'timestamp',
        max_length: int = 64,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> None:
        super(TestCausalDataset, self).__init__(
            interactions=interactions,
            ground_truth=ground_truth,
            add_labels=add_labels,
            user_column=user_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            max_length=max_length,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
        )


class TrainCausalDataset(_TrainMixin, _CausalMixin, _SequentialDataset):
    def __init__(
        self,
        interactions: pl.DataFrame,
        user_column: str = 'user_id',
        item_column: str = 'item_id',
        timestamp_column: str = 'timestamp',
        max_length: int = 64,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> None:
        super(TrainCausalDataset, self).__init__(
            interactions=interactions,
            user_column=user_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            max_length=max_length,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
        )
