import torch
from torch.optim import Adam


class ConstrainedNormAdam(Adam):
    """Adam with post-step norm clipping."""

    def __init__(
        self,
        params,
        constrained_params,
        pad_token_id: int = 0,
        max_norm: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.constrained_params = next(constrained_params)
        self.pad_token_id = pad_token_id
        self.max_norm = max_norm

        self.mask = torch.ones(self.constrained_params.data.shape[0], dtype=bool)
        self.mask[pad_token_id] = False

    def step(self, closure=None) -> None:
        super().step(closure=closure)
        p = self.constrained_params

        with torch.no_grad():
            p[self.mask] = torch.renorm(p[self.mask], 2, 0, self.max_norm)
