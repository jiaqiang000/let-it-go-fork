import torch
import torch.nn as nn


class _RecommenderModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        intermediate_dim: int,
        p: float,
        max_length: int,
        init_range: float,
        pad_token_id: int,
    ) -> None:
        super(_RecommenderModel, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.p = p
        self.max_length = max_length
        self.init_range = init_range
        self.pad_token_id = pad_token_id

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        inputs_embeddings = self.item_embedding(inputs)
        output = self._forward(inputs_embeddings, padding_mask)
        return torch.matmul(output, self.item_embedding.weight.T)

    def _forward(self, inputs_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _freeze_padding_embedding_hook(self, grad: torch.Tensor) -> torch.Tensor:
        grad[self.pad_token_id] = 0.0
        return grad

    def _add_padding_embedding(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.vstack(
            (
                item_embeddings[: self.pad_token_id],
                torch.zeros(self.embedding_dim),
                item_embeddings[self.pad_token_id :],
            )
        )

    def set_pretrained_item_embeddings(
        self,
        item_embeddings: torch.Tensor,
        add_padding_embedding: bool = True,
        freeze: bool = False,
    ) -> None:
        if add_padding_embedding:
            item_embeddings = self._add_padding_embedding(item_embeddings)

        self.item_embedding = nn.Embedding.from_pretrained(
            item_embeddings, freeze=freeze, padding_idx=self.pad_token_id
        )
