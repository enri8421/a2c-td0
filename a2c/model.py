import math

import torch


class OrthogonalLinear(torch.nn.Linear):
    """Linear layer with orthogonal weights and zero bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        gain: float = math.sqrt(2),
    ) -> None:
        super().__init__(in_features, out_features)
        torch.nn.init.orthogonal_(self.weight, gain=gain)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class SimpleNN(torch.nn.Module):
    """Two 64-unit Tanh layers for policy logits or state values."""

    def __init__(
        self,
        observation_size: int,
        n_output: int,
        *,
        output_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.ffl = torch.nn.Sequential(
            OrthogonalLinear(observation_size, 64),
            torch.nn.Tanh(),
            OrthogonalLinear(64, 64),
            torch.nn.Tanh(),
            OrthogonalLinear(64, n_output, gain=output_gain),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map (..., observation_size) to (..., n_output)."""
        return self.ffl(observations)
