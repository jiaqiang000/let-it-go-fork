import torch

from ._model import _RecommenderModel


class HistoryAverageModel(_RecommenderModel):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_length: int = 64,
        pad_token_id: int = 0,
    ) -> None:
        super(HistoryAverageModel, self).__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_blocks=None,
            num_heads=None,
            intermediate_dim=None,
            p=None,
            max_length=max_length,
            init_range=None,
            pad_token_id=pad_token_id,
        )

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'item_embedding'):
            msg = 'Model is missing item embeddings. Call `set_pretrained_item_embeddings` first.'
            raise RuntimeError(msg)

        return super(HistoryAverageModel, self).forward(inputs, padding_mask).unsqueeze(1)

    def _forward(self, inputs_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return inputs_embeddings.sum(dim=1) / padding_mask.sum(dim=1).unsqueeze(1)
