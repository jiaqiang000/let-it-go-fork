from typing import List

import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class _SequentialDataset(Dataset):
    def __init__(
        self,
        interactions: pl.DataFrame,
        add_labels: bool,
        user_column: str,
        item_column: str,
        timestamp_column: str,
        max_length: int,
        pad_token_id: int,
        ignore_index: int,
    ) -> None:
        self.add_labels = add_labels
        self.user_column = user_column
        self.item_column = item_column
        self.timestamp_column = timestamp_column
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        self.interactions = self._make_sequential(interactions)

        if add_labels:
            self.interactions = self.interactions.filter(pl.col('history').list.len() > 1)

    def __getitem__(self, index: int) -> dict:
        item = self.interactions.row(index, named=True)
        item['inputs'] = torch.tensor(item['history'][-self.max_length :])

        if 'labels' in item:
            item['labels'] = torch.tensor(item['labels'])

        return item

    def __len__(self) -> int:
        return len(self.interactions)

    def _make_sequential(self, interactions: pl.DataFrame) -> pl.DataFrame:
        return (
            interactions.sort(self.timestamp_column)
            .group_by(self.user_column, maintain_order=True)
            .agg(pl.col(self.item_column).alias('history'))
        )

    def _create_padding_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs != self.pad_token_id).float()

    def collate_fn(self, batch: List[dict]) -> dict:
        collated_batch = {}

        for key in batch[0].keys():
            collated_batch[key] = [item[key] for item in batch]

        collated_batch['inputs'] = pad_sequence(
            collated_batch['inputs'],
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side='left',
        )
        collated_batch['padding_mask'] = self._create_padding_mask(collated_batch['inputs'])

        if 'labels' in collated_batch:
            if collated_batch['labels'][0].dim() == 0:
                collated_batch['labels'] = torch.stack(collated_batch['labels'])
            else:
                collated_batch['labels'] = pad_sequence(
                    collated_batch['labels'],
                    batch_first=True,
                    padding_value=self.ignore_index,
                    padding_side='left',
                )

        return collated_batch


class _TestMixin:
    def __init__(self, ground_truth: pl.DataFrame | None, **kwargs) -> None:
        super(_TestMixin, self).__init__(**kwargs)

        if ground_truth is not None:
            if self.add_labels:
                msg = 'Argument `add_labels` must be `False` when `ground_truth` is provided.'
                raise ValueError(msg)

            self.interactions = self.interactions.join(
                ground_truth.rename({self.item_column: 'labels'}), on=self.user_column
            )

    def __getitem__(self, index: int) -> dict:
        item = super(_TestMixin, self).__getitem__(index)

        if self.add_labels:
            item['labels'] = item['labels'][-1]
            item['history'] = item['history'][:-1]

        return item


class _TrainMixin:
    def __init__(self, **kwargs) -> None:
        super(_TrainMixin, self).__init__(add_labels=True, **kwargs)
