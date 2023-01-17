from typing import Callable, Tuple

from gym import spaces
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


class CardEmbeddingExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Space,
                 features_dim: int = 16,
                 layers: int = 3,
                 activation_fn: nn.Module = nn.ELU
                 ):
        super().__init__(observation_space, features_dim=features_dim)

        assert layers >= 0

        if layers == 0:
            self.card_embedding = nn.Sequential(
                nn.Identity()
            )
            self.set_embedding = nn.Sequential(
                nn.Identity()
            )
        else:
            self.card_embedding = nn.Sequential()
            self.set_embedding = nn.Sequential()

            for _ in range(layers):
                self.card_embedding.extend([nn.Linear(features_dim, features_dim), activation_fn()])
                self.set_embedding.extend([nn.Linear(features_dim, features_dim), activation_fn()])

    def forward(self, observations) -> Tuple[th.Tensor, th.Tensor]:
        number_of_envs = observations.shape[0]
        number_of_cards = observations.shape[1] // 16

        observations = observations.view([number_of_envs * number_of_cards, 16])

        embed_cards = self.card_embedding(observations)
        embed_cards = embed_cards.view([number_of_envs, number_of_cards, 16])

        embed_set = self.set_embedding(embed_cards.sum(dim=1))
        embed_set = embed_set.view([number_of_envs, 1, 16])

        return embed_cards, embed_set


class PermutationEquivariantNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int = 16,
        hidden_layer_dim: int = 81,
        layers: int = 1,
        last_layer_dim_pi: int = 3,
        last_layer_dim_vf: int = 1,
        activation_fn: nn.Module = nn.ELU
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.feature_dim = feature_dim
        self.latent_dim = hidden_layer_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential()

        for i in range(layers + 1):
            input_dim = feature_dim * 2 if i == 0 else hidden_layer_dim
            output_dim = 1 if i == layers else hidden_layer_dim

            self.policy_net.extend([nn.Linear(input_dim, output_dim), activation_fn()])

        # Value network
        self.value_net = nn.Sequential()

        for i in range(layers + 1):
            input_dim = feature_dim if i == 0 else hidden_layer_dim
            output_dim = last_layer_dim_vf if i == layers else hidden_layer_dim

            self.value_net.extend([nn.Linear(input_dim, output_dim), activation_fn()])

    def forward(self, features: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        embed_cards, embed_set = features  # (n_envs, n_cards, 16), (n_envs, 1, 16)

        number_of_envs = embed_cards.shape[0]
        number_of_cards = embed_cards.shape[1]

        # Concat the embed_set ("context vector") to every card in embed_cards
        features = th.cat([embed_cards, embed_set.expand(-1, number_of_cards, -1)], dim=2)  # (n_envs, n_cards, 32)
        features = features.view([number_of_envs * number_of_cards, 32])  # (n_envs * n_cards, 32)

        outputs = self.policy_net(features)  # (n_envs * n_cards, 1)
        outputs = outputs.view((number_of_envs, number_of_cards))  # (n_envs, n_cards)
        outputs = outputs.softmax(dim=1)  # (n_envs, n_cards)

        return outputs

    def forward_critic(self, features: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        _, embed_set = features  # _, (n_envs, 1, 16)

        outputs = self.value_net(embed_set)  # (n_envs, 1)

        return outputs


class PermutationEquivariantPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            features_extractor_class=CardEmbeddingExtractor,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PermutationEquivariantNetwork(self.features_dim)

