"""
Simple (MLP) feature extractor for LOCM.

Concatenates all observations into a flat vector and passes through an MLP.
Only compatible with SimpleLOCMNetwork actor-critic.
"""

import torch as th
import torch.nn as nn
import gymnasium as gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SimpleFeaturesExtractor(BaseFeaturesExtractor):
    """Flat MLP feature extractor — concatenates all observations into a single vector."""

    def __init__(
        self,
        observation_space: gym.Space,
        net_arch: list[int],
        activation_fn: type[nn.Module],
        card_dim: int = 17,
        player_dim: int = 5,
        creature_dim: int = 17,
    ):
        input_dim = (
            2 * player_dim  # players
            + card_dim  # deck
            + 8 * card_dim  # hand cards
            + 4 * 3 * creature_dim  # lane creatures
            # = 367
        )

        super().__init__(observation_space, features_dim=net_arch[-1])

        net_arch = [input_dim] + net_arch

        layers = []

        for i in range(len(net_arch) - 1):
            layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
            layers.append(activation_fn())

        self.fc = nn.Sequential(*layers)

    def forward(self, features: dict) -> dict[str, th.Tensor]:
        player_stats = features["player_stats"]  # [bs, 5]
        opponent_stats = features["opponent_stats"]  # [bs, 5]
        player_deck = features["player_deck"]  # [bs, 30, 17]
        player_hand = features["player_hand"]  # [bs, 8, 17]
        p_lane0 = features["player_lane0"]  # [bs, 3, 17]
        p_lane1 = features["player_lane1"]  # [bs, 3, 17]
        op_lane0 = features["opponent_lane0"]  # [bs, 3, 17]
        op_lane1 = features["opponent_lane1"]  # [bs, 3, 17]

        bs = player_stats.shape[0]

        deck_mean = player_deck.mean(dim=1)  # [bs, 17]
        hand_flat = player_hand.view(bs, -1)  # [bs, 8*17]
        p_lane0_flat = p_lane0.view(bs, -1)  # [bs, 3*17]
        p_lane1_flat = p_lane1.view(bs, -1)  # [bs, 3*17]
        op_lane0_flat = op_lane0.view(bs, -1)  # [bs, 3*17]
        op_lane1_flat = op_lane1.view(bs, -1)  # [bs, 3*17]

        features_concat = th.cat((
            player_stats,
            opponent_stats,
            deck_mean,
            hand_flat,
            p_lane0_flat,
            p_lane1_flat,
            op_lane0_flat,
            op_lane1_flat,
        ), dim=1)

        return dict(latent=self.fc(features_concat), action_mask=features["action_mask"])
