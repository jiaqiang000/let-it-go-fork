import torch
import torch.nn as nn

from ...recommender.sasrec import SASRecModel


class _TrainableDeltaMixin:
    def __init__(self, max_delta_norm: float, **kwargs) -> None:
        super(_TrainableDeltaMixin, self).__init__(**kwargs)
        self.max_delta_norm = max_delta_norm
        self.delta_embedding = nn.Embedding(
            self.num_items + 1,
            self.embedding_dim,
            padding_idx=self.pad_token_id,
            max_norm=max_delta_norm,
        )
        self.delta_embedding.weight.register_hook(self._freeze_padding_embedding_hook)
        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        inputs_embeddings = self.item_embedding(inputs) + self.delta_embedding(inputs)
        output = self._forward(inputs_embeddings, padding_mask)
        return torch.matmul(output, (self.item_embedding.weight + self.delta_embedding.weight).T)

    def set_pretrained_item_embeddings(
        self,
        item_embeddings: torch.Tensor,
        delta_embeddings: torch.Tensor | None = None,
        add_padding_embedding: bool = True,
        freeze: bool = False,  # Ignored
    ) -> None:
        super(_TrainableDeltaMixin, self).set_pretrained_item_embeddings(
            item_embeddings,
            add_padding_embedding=add_padding_embedding,
            freeze=True,
        )

        if delta_embeddings is not None:
            if item_embeddings.shape != delta_embeddings.shape:
                msg = (
                    'Tensors `item_embeddings` and `delta_embeddings` must have the same shape. '
                    f'Found {item_embeddings.shape} and {delta_embeddings.shape}.'
                )
                raise ValueError(msg)

            self.delta_embedding = nn.Embedding.from_pretrained(
                delta_embeddings,
                freeze=True,
                padding_idx=self.pad_token_id,
            )


class SASRecModelWithTrainableDelta(_TrainableDeltaMixin, SASRecModel):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        intermediate_dim: int = 128,
        p: float = 0.1,
        max_length: int = 64,
        init_range: float = 0.02,
        pad_token_id: int = 0,
        max_delta_norm: float = 0.5,
    ) -> None:
        super(SASRecModelWithTrainableDelta, self).__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            p=p,
            max_length=max_length,
            init_range=init_range,
            pad_token_id=pad_token_id,
            max_delta_norm=max_delta_norm,
        )
