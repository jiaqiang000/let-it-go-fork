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


class _QualityAwareTrainableDeltaMixin:
    def __init__(self, delta_budget: torch.Tensor, **kwargs) -> None:
        super(_QualityAwareTrainableDeltaMixin, self).__init__(**kwargs)
        self.delta_embedding = nn.Embedding(
            self.num_items + 1,
            self.embedding_dim,
            padding_idx=self.pad_token_id,
        )
        self.delta_embedding.weight.register_hook(self._freeze_padding_embedding_hook)
        self.apply(self._init_weights)
        self.set_delta_budget(delta_budget, add_padding_budget=True)

    def _add_padding_budget(self, delta_budget: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                delta_budget[: self.pad_token_id],
                torch.zeros(1, dtype=delta_budget.dtype, device=delta_budget.device),
                delta_budget[self.pad_token_id :],
            )
        )

    def set_delta_budget(self, delta_budget: torch.Tensor, add_padding_budget: bool = True) -> None:
        delta_budget = delta_budget.detach().clone().float()

        if delta_budget.ndim != 1:
            raise ValueError(
                f"`delta_budget` must be 1D, found shape {tuple(delta_budget.shape)}."
            )

        if add_padding_budget:
            delta_budget = self._add_padding_budget(delta_budget)

        expected_size = self.item_embedding.weight.shape[0]
        if delta_budget.shape[0] != expected_size:
            raise ValueError(
                f"`delta_budget` length must match item embeddings ({expected_size}), "
                f"found {delta_budget.shape[0]}."
            )

        if "delta_budget" in self._buffers:
            self.delta_budget = delta_budget
        else:
            self.register_buffer("delta_budget", delta_budget)

    @staticmethod
    def _project_delta(delta_raw: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        norm = delta_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        safe_budget = budget.clamp_min(0.0)
        scale = torch.minimum(torch.ones_like(norm), safe_budget / norm)
        return delta_raw * scale

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        delta = self._project_delta(
            self.delta_embedding(inputs),
            self.delta_budget[inputs].unsqueeze(-1),
        )
        inputs_embeddings = self.item_embedding(inputs) + delta
        output = self._forward(inputs_embeddings, padding_mask)

        all_delta = self._project_delta(
            self.delta_embedding.weight,
            self.delta_budget.unsqueeze(-1),
        )
        return torch.matmul(output, (self.item_embedding.weight + all_delta).T)

    def set_pretrained_item_embeddings(
        self,
        item_embeddings: torch.Tensor,
        delta_embeddings: torch.Tensor | None = None,
        delta_budget: torch.Tensor | None = None,
        add_padding_embedding: bool = True,
        freeze: bool = False,  # Ignored
    ) -> None:
        super(_QualityAwareTrainableDeltaMixin, self).set_pretrained_item_embeddings(
            item_embeddings,
            add_padding_embedding=add_padding_embedding,
            freeze=True,
        )

        if delta_embeddings is not None:
            if item_embeddings.shape != delta_embeddings.shape:
                msg = (
                    "Tensors `item_embeddings` and `delta_embeddings` must have the same shape. "
                    f"Found {item_embeddings.shape} and {delta_embeddings.shape}."
                )
                raise ValueError(msg)

            self.delta_embedding = nn.Embedding.from_pretrained(
                delta_embeddings,
                freeze=True,
                padding_idx=self.pad_token_id,
            )

        if delta_budget is not None:
            self.set_delta_budget(delta_budget, add_padding_budget=add_padding_embedding)


class SASRecModelWithQualityAwareTrainableDelta(_QualityAwareTrainableDeltaMixin, SASRecModel):
    def __init__(
        self,
        num_items: int,
        delta_budget: torch.Tensor,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        intermediate_dim: int = 128,
        p: float = 0.1,
        max_length: int = 64,
        init_range: float = 0.02,
        pad_token_id: int = 0,
    ) -> None:
        super(SASRecModelWithQualityAwareTrainableDelta, self).__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            p=p,
            max_length=max_length,
            init_range=init_range,
            pad_token_id=pad_token_id,
            delta_budget=delta_budget,
        )
