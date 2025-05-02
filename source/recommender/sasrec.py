import torch
import torch.nn as nn

from ._model import _RecommenderModel


class _FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim: int, intermediate_dim: int, p: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(intermediate_dim, embedding_dim),
            nn.Dropout(p=p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SASRecBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, num_heads: int, intermediate_dim: int, p: float, max_length: int
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim, eps=1e-8)
        self.layer_norm2 = nn.LayerNorm(embedding_dim, eps=1e-8)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=p, batch_first=True)
        self.ffn = _FeedForwardNetwork(embedding_dim, intermediate_dim, p)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        q = self.layer_norm1(x)

        causal_mask = torch.triu(
            torch.full((x.shape[1], x.shape[1]), -torch.inf, device=x.device), diagonal=1
        )
        attn_out, _ = self.attn(q, x, x, key_padding_mask=padding_mask, attn_mask=causal_mask)

        x = q + attn_out
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x


class SASRecModel(_RecommenderModel):
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
    ) -> None:
        super(SASRecModel, self).__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            p=p,
            max_length=max_length,
            init_range=init_range,
            pad_token_id=pad_token_id,
        )
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=pad_token_id)
        self.item_embedding.weight.register_hook(self._freeze_padding_embedding_hook)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                _SASRecBlock(embedding_dim, num_heads, intermediate_dim, p, max_length)
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-8)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_range)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.init_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _forward(self, inputs_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = inputs_embeddings

        x *= self.embedding_dim**0.5
        x += self.pos_embedding(torch.arange(x.shape[1], dtype=torch.long, device=x.device))
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x, padding_mask)

        return self.layer_norm(x)
