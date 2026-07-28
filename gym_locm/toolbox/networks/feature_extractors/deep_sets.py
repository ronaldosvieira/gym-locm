"""
Deep Sets feature extractor for LOCM.

Permutation-invariant architecture that embeds each entity through shared MLPs,
then aggregates via sum-pooling over set dimensions (Zaheer et al., 2017).
"""

import torch as th
import torch.nn as nn
import gymnasium as gym

from gym_locm.toolbox.networks.feature_extractors.base import LOCMFeaturesExtractor


class DeepSetsFeaturesExtractor(LOCMFeaturesExtractor):
    """Deep Sets feature extractor with per-entity MLPs and sum-pooling."""

    def __init__(
        self,
        observation_space: gym.Space,
        card_dim: int = 17,
        player_dim: int = 5,
        creature_dim: int = 17,
        card_emb_dim: int = 32,
        zone_emb_dim: int = 32,
        player_emb_dim: int = 16,
        creature_emb_dim: int = 32,
        lane_emb_dim: int = 64,
        state_emb_dim: int = 128,
    ):
        state_input_dim = (
            2 * player_emb_dim
            + 2 * zone_emb_dim
            + 4 * lane_emb_dim
        )

        super().__init__(
            observation_space,
            features_dim=state_emb_dim
        )  # features_dim is used to calculate LSTM input size

        self._card_emb_dim = card_emb_dim
        self._zone_emb_dim = zone_emb_dim
        self._creature_emb_dim = creature_emb_dim
        self._lane_emb_dim = lane_emb_dim
        self._state_emb_dim = state_emb_dim

        self.player_embedding = nn.Sequential(
            nn.Linear(player_dim, player_emb_dim), nn.ReLU(),
            nn.Linear(player_emb_dim, player_emb_dim), nn.ReLU(),
        ) # 5 * 16 + 16 * 16 = 336 parameters

        self.card_embedding = nn.Sequential(
            nn.Linear(card_dim, card_emb_dim), nn.ReLU(),
            nn.Linear(card_emb_dim, card_emb_dim), nn.ReLU(),
        ) # 17 * 32 + 32 * 32 = 1,568 parameters

        self.card_zone_embedding = nn.Sequential(
            nn.Linear(card_emb_dim, zone_emb_dim), nn.ReLU(),
        ) # 32 * 32 = 1,024 parameters

        self.creature_embedding = nn.Sequential(
            nn.Linear(creature_dim, creature_emb_dim), nn.ReLU(),
            nn.Linear(creature_emb_dim, creature_emb_dim), nn.ReLU(),
        ) # 8 * 16 + 16 * 16 = 384 parameters

        self.lane_embedding = nn.Sequential(
            nn.Linear(creature_emb_dim, lane_emb_dim), nn.ReLU(),
        ) # 16 * 16 = 256 parameters

        # 2 * 16 player + 32 hand + 32 deck + 4 * 16 lane = 160 features
        self.state_embedding = nn.Sequential(
            nn.Linear(state_input_dim, state_emb_dim), nn.ReLU(),
            nn.Linear(state_emb_dim, state_emb_dim), nn.ReLU(),
        ) # 160 * 256 + 256 * 256 = 106,496 parameters

    @property
    def hand_cards_dim(self) -> int:
        return self._zone_emb_dim

    @property
    def creature_tokens_dim(self) -> int:
        return self._creature_emb_dim

    @property
    def lane_dim(self) -> int:
        return self._lane_emb_dim

    @property
    def state_dim(self) -> int:
        return self._state_emb_dim

    def forward(self, observations) -> dict[str, th.Tensor]:
        # embedding of both players
        p = self.player_embedding(observations["player_stats"])
        op = self.player_embedding(observations["opponent_stats"])

        p_deck_cards = observations["player_deck"]

        # embedding of individual deck cards
        p_deck_cards = self.card_embedding(p_deck_cards)

        # embedding of the whole deck
        p_deck = p_deck_cards.sum(dim=1)
        p_deck = self.card_zone_embedding(p_deck)

        p_hand_cards = observations["player_hand"]

        # embedding of individual hand cards
        p_hand_cards = self.card_embedding(p_hand_cards)

        # embedding of the whole hand
        p_hand = p_hand_cards.sum(dim=1)
        p_hand = self.card_zone_embedding(p_hand)

        p_lane0_creatures = observations["player_lane0"]

        # embedding of individual player lane 0 creatures
        p_lane0_creatures = self.creature_embedding(p_lane0_creatures)

        # embedding of the whole player lane 0
        p_lane0 = p_lane0_creatures.sum(dim=1)
        p_lane0 = self.lane_embedding(p_lane0)

        p_lane1_creatures = observations["player_lane1"]

        # embedding of individual player lane 1 creatures
        p_lane1_creatures = self.creature_embedding(p_lane1_creatures)

        # embedding of the whole player lane 1
        p_lane1 = p_lane1_creatures.sum(dim=1)
        p_lane1 = self.lane_embedding(p_lane1)

        op_lane0_creatures = observations["opponent_lane0"]

        # embedding of individual opponent lane 0 creatures
        op_lane0_creatures = self.creature_embedding(op_lane0_creatures)

        # embedding of the whole opponent lane 0
        op_lane0 = op_lane0_creatures.sum(dim=1)
        op_lane0 = self.lane_embedding(op_lane0)

        op_lane1_creatures = observations["opponent_lane1"]

        # embedding of individual opponent lane 1 creatures
        op_lane1_creatures = self.creature_embedding(op_lane1_creatures)

        # embedding of the whole opponent lane 1
        op_lane1 = op_lane1_creatures.sum(dim=1)
        op_lane1 = self.lane_embedding(op_lane1)

        # embedding of the whole state
        state_input = th.cat((
            p, op,
            p_deck, p_hand,
            p_lane0, p_lane1,
            op_lane0, op_lane1
        ), dim=1)
        state = self.state_embedding(state_input)

        embeddings = dict(
            player=p,
            opponent=op,
            deck_cards=p_deck_cards,
            deck=p_deck,
            hand_cards=p_hand_cards,
            hand=p_hand,
            p_lane0_creatures=p_lane0_creatures,
            p_lane0=p_lane0,
            p_lane1_creatures=p_lane1_creatures,
            p_lane1=p_lane1,
            op_lane0_creatures=op_lane0_creatures,
            op_lane0=op_lane0,
            op_lane1_creatures=op_lane1_creatures,
            op_lane1=op_lane1,
            state=state,
            action_mask=observations["action_mask"],
        )

        return embeddings
