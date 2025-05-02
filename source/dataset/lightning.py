import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .causal import TestCausalDataset, TrainCausalDataset
from ._utils import load_data


class SequentialDataModule(LightningDataModule):
    def __init__(
        self,
        train_filepath: str,
        val_filepath: str,
        user_column: str = 'user_id',
        item_column: str = 'item_id',
        timestamp_column: str = 'timestamp',
        max_length: int = 64,
        batch_size: int = 128,
        num_workers: int = 128,
    ) -> None:
        super().__init__()

        if os.path.exists(train_filepath):
            self.train_filepath = train_filepath
        else:
            msg = f'Failed to find training data at path: {train_filepath}.'
            raise FileNotFoundError(msg)

        if os.path.exists(val_filepath):
            self.val_filepath = val_filepath
        else:
            msg = f'Failed to find validation data at path: {val_filepath}.'
            raise FileNotFoundError(msg)

        self.dataset_params = {
            'user_column': user_column,
            'item_column': item_column,
            'timestamp_column': timestamp_column,
            'max_length': max_length,
        }
        self.dataloader_params = {
            'batch_size': batch_size,
            'num_workers': num_workers,
        }

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_dataset = TrainCausalDataset(
                load_data(self.train_filepath),
                **self.dataset_params,
            )
            self.val_dataset = TestCausalDataset(
                load_data(self.val_filepath),
                add_labels=True,
                **self.dataset_params,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            **self.dataloader_params,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            **self.dataloader_params,
        )
