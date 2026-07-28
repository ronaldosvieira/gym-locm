"""
Base class for LOCM actor-critic networks.

All actor-critic networks receive entity embeddings from a feature extractor
and produce:
- 145-dim action logits (policy)
- 1-dim value estimate (critic)
"""

import torch as th
import torch.nn as nn
from abc import abstractmethod


class LOCMActorCriticNetwork(nn.Module):
    """Base class for LOCM actor-critic networks (mlp_extractor role in SB3)."""

    def __init__(self, last_layer_dim_pi: int = 145, last_layer_dim_vf: int = 1):
        super().__init__()
        # Required by SB3 to create distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def forward(self, features: dict) -> tuple[th.Tensor, th.Tensor]:
        """
        :return: (latent_policy, latent_value) tensors.
        """
        return self.forward_actor(features), self.forward_critic(features)

    @abstractmethod
    def forward_actor(self, features: dict) -> th.Tensor:
        """Compute masked action logits [bs, 145]."""
        ...

    @abstractmethod
    def forward_critic(self, features: dict) -> th.Tensor:
        """Compute value estimate [bs, 1]."""
        ...

    @abstractmethod
    def get_policy_modules(self) -> list[nn.Module]:
        """Return modules that should receive policy init gain (0.01)."""
        ...

    @abstractmethod
    def get_value_modules(self) -> list[nn.Module]:
        """Return modules that should receive value init gain (1.0)."""
        ...
