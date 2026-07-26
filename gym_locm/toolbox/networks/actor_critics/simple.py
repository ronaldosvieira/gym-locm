import torch as th
import torch.nn as nn
from typing import Tuple, List

from gym_locm.toolbox.networks.actor_critics.base import LOCMActorCriticNetwork


class SimpleLOCMNetwork(LOCMActorCriticNetwork):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 145,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        self.policy = nn.Linear(feature_dim, last_layer_dim_pi)
        self.value_function = nn.Linear(feature_dim, last_layer_dim_vf)

    def get_policy_modules(self) -> List[nn.Module]:
        return [self.policy]

    def get_value_modules(self) -> List[nn.Module]:
        return [self.value_function]

    def forward_actor(self, features: dict) -> th.Tensor:
        logits = self.policy(features.get("latent"))

        action_mask = features.get("action_mask")
        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits

    def forward_critic(self, features: dict) -> th.Tensor:
        return self.value_function(features.get("latent"))
