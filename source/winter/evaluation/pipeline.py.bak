import logging

import polars as pl
from lightning import Trainer
from torch.utils.data import DataLoader

from ...dataset import TestCausalDataset
from .metrics import ColdStartOfflineMetrics
from ..recommender import ColdStartSequentialRecommender


class ColdStartEvaluationPipeline:
    def __init__(
        self,
        recommender: ColdStartSequentialRecommender,
        trainer: Trainer,
        interactions: pl.DataFrame,
        ground_truth: pl.DataFrame,
        user_column: str = 'user_id',
        item_column: str = 'item_id',
        rating_column: str = 'rating',
        timestamp_column: str = 'timestamp',
        cold_flag_column: str = 'is_cold',
        batch_size: int = 128,
        num_workers: int = 128,
    ) -> None:
        self.recommender = recommender
        self.trainer = trainer
        self.interactions = interactions
        self.ground_truth = ground_truth
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.timestamp_column = timestamp_column
        self.cold_flag_column = cold_flag_column
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.evaluator = ColdStartOfflineMetrics(
            metrics=recommender.metrics,
            topk=recommender.topk,
            user_column=user_column,
            item_column=item_column,
            rating_column=rating_column,
            cold_flag_column=cold_flag_column,
        )

        if len(recommender.model.item_embedding.weight) > recommender.model.num_items + 1:
            self.cold_items_available = True
            logging.info('ColdStartEvaluationPipeline: cold items are available.')
        else:
            self.cold_items_available = False
            logging.info('ColdStartEvaluationPipeline: cold items are missed.')

    def _build_dataloader(self, interactions: pl.DataFrame) -> DataLoader:
        dataset = TestCausalDataset(
            interactions,
            add_labels=False,
            user_column=self.user_column,
            item_column=self.item_column,
            timestamp_column=self.timestamp_column,
            max_length=self.recommender.model.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
        )

    def run(self) -> pl.DataFrame:
        results = []

        for recommend_cold_items in (True, False):
            for filter_cold_items in (True, False):
                if not self.cold_items_available:
                    if recommend_cold_items:
                        logging.info('ColdStartEvaluationPipeline: skip experiment.')
                        continue

                    if not filter_cold_items:
                        logging.info('ColdStartEvaluationPipeline: skip experiment.')
                        continue

                dataloader = self._build_dataloader(
                    self.interactions.filter(~pl.col(self.cold_flag_column))
                    if filter_cold_items
                    else self.interactions
                )
                self.recommender.recommend_cold_items = recommend_cold_items
                recommendations = pl.concat(
                    self.trainer.predict(self.recommender, dataloaders=dataloader)
                )

                # Fix dtype
                recommendations = recommendations.with_columns(
                    pl.col(self.user_column).cast(self.ground_truth.schema[self.user_column])
                )

                result = self.evaluator(recommendations, self.ground_truth)
                result['recommend-cold-items'] = recommend_cold_items
                result['filter-cold-items'] = filter_cold_items

                results.append(result)

        return pl.from_dicts(results)
