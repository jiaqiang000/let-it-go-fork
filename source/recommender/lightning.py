from typing import Dict, List, Tuple, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from lightning import LightningModule
from replay.metrics import MRR, NDCG, Recall, OfflineMetrics

from ._model import _RecommenderModel


class SequentialRecommender(LightningModule):
    AVAILABLE_METRICS = {
        'MRR': MRR,
        'NDCG': NDCG,
        'Recall': Recall,
    }

    def __init__(
        self,
        model: _RecommenderModel,
        learning_rate: float = 1e-3,
        remove_seen: bool = True,
        metrics: List[str] | str = ['MRR', 'NDCG', 'Recall'],
        topk: List[int] | int = [5, 10, 20],
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.remove_seen = remove_seen
        self.metrics = metrics
        self.topk = topk

        if isinstance(metrics, str):
            metrics = [metrics]

        self._metrics = []

        for metric in metrics:
            if metric in self.AVAILABLE_METRICS:
                self._metrics.append(self.AVAILABLE_METRICS[metric](topk))

        self._metrics = OfflineMetrics(self._metrics, query_column='user_id')
        self.k_max = max(topk) if isinstance(topk, list) else topk
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.model(inputs, padding_mask)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch['inputs'], batch['padding_mask'])
        loss = self.loss_fn(logits.view(-1, self.model.num_items + 1), batch['labels'].view(-1))
        self.log('recommender-train-loss', loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch['inputs'], batch['padding_mask'])[:, -1, :]
        loss = self.loss_fn(logits, batch['labels'])
        self.log('recommender-val-loss', loss, prog_bar=True, on_epoch=True)

        scores, items = self._recommend(logits, batch['history'], self.k_max)
        recommendations = self._convert_to_polars(
            {'user_id': batch['user_id'], 'item_id': items, 'rating': scores}
        )
        ground_truth = self._convert_to_polars(
            {'user_id': batch['user_id'], 'item_id': batch['labels']}
        )
        self._compute_metrics(recommendations, ground_truth)

    def predict_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch['inputs'], batch['padding_mask'])[:, -1, :]
        scores, items = self._recommend(logits, batch['history'], self.k_max)
        return self._convert_to_polars(
            {'user_id': batch['user_id'], 'item_id': items, 'rating': scores}
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _recommend(
        self, logits: torch.Tensor, history: List[List[int]], k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.remove_seen:
            for i, seen in enumerate(history):
                logits[i, seen] = -torch.inf

        scores, items = torch.sort(logits, descending=True)
        scores, items = scores[:, :k], items[:, :k]

        return scores, items

    def _compute_metrics(self, recommendations: pl.DataFrame, ground_truth: pl.DataFrame) -> None:
        metrics = self._metrics(recommendations, ground_truth)
        metrics = {f'recommender-val-{key}': value for key, value in metrics.items()}
        self.log_dict(metrics, on_epoch=True)

    def set_pretrained_item_embeddings(
        self,
        item_embeddings: torch.Tensor,
        add_padding_embedding: bool = True,
        freeze: bool = False,
    ) -> None:
        self.model.set_pretrained_item_embeddings(
            item_embeddings,
            add_padding_embedding=add_padding_embedding,
            freeze=freeze,
        )

    @staticmethod
    def _convert_to_polars(data: Dict[str, Union[list, torch.Tensor]]) -> pl.DataFrame:
        repeats, flattened = None, set()

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 2:
                    repeats = value.shape[1]
                    value = value.flatten()
                    flattened.add(key)

                data[key] = value.tolist()

        if repeats:
            for key, value in data.items():
                if key not in flattened:
                    data[key] = np.repeat(value, repeats)

        return pl.DataFrame(data)
