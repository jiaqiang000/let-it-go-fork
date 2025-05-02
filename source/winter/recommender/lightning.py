from typing import List, Tuple

import torch

from ...recommender.lightning import SequentialRecommender
from ...recommender._model import _RecommenderModel


class ColdStartSequentialRecommender(SequentialRecommender):
    def __init__(
        self,
        model: _RecommenderModel,
        learning_rate: float = 1e-3,
        remove_seen: bool = True,
        metrics: List[str] | str = ['MRR', 'NDCG', 'Recall'],
        topk: List[int] | int = [5, 10, 20],
        recommend_cold_items: bool = False,
    ) -> None:
        super().__init__(
            model, learning_rate=learning_rate, remove_seen=remove_seen, metrics=metrics, topk=topk
        )
        self.recommend_cold_items = recommend_cold_items

    def _recommend(
        self, logits: torch.Tensor, history: List[List[int]], k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.remove_seen:
            for i, seen in enumerate(history):
                logits[i, seen] = -torch.inf

        if not self.recommend_cold_items:
            logits = logits[:, : self.model.num_items + 1]

        scores, items = torch.sort(logits, descending=True)
        scores, items = scores[:, :k], items[:, :k]

        return scores, items
