import torch as th
import torch.nn as nn
from typing import Tuple

from gym_locm.toolbox.networks.feature_extractors.base import LOCMFeaturesExtractor
from gym_locm.toolbox.networks.attention import SAB, PMA

class SetTransformer2FeaturesExtractor(LOCMFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        card_dim: int = 17,
        player_dim: int = 5,
        creature_dim: int = 8,
        card_emb_dim: int = 32,
        zone_emb_dim: int = 32,
        player_emb_dim: int = 16,
        creature_emb_dim: int = 16,
        lane_emb_dim: int = 16,
        state_emb_dim: int = 256,
        **kwargs,
    ):
        super().__init__(observation_space, features_dim=64, **kwargs)

        self._emb_dim = 64

        self.player_embedding = nn.Sequential(
            nn.Linear(player_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        self.card_embedding = nn.Sequential(
            nn.Linear(card_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        
        self.deck_embedding = nn.Sequential(
            SAB(32, 32, num_heads=4),
            SAB(32, 32, num_heads=4),
            PMA(dim=32, num_heads=4, num_seeds=1)
        )

        self.global_embedding = nn.Sequential(
            SAB(40, 64, num_heads=4),
            SAB(64, 64, num_heads=4),
            SAB(64, 64, num_heads=4),
        )

        self.lane_pooling = PMA(dim=64, num_heads=4, num_seeds=1, ln=True)
        self.state_pooling = PMA(dim=64, num_heads=4, num_seeds=1, ln=True)

    @property
    def card_emb_dim(self) -> int:
        return self._emb_dim

    @property
    def creature_emb_dim(self) -> int:
        return self._emb_dim

    @property
    def lane_emb_dim(self) -> int:
        return self._emb_dim

    @property
    def state_emb_dim(self) -> int:
        return self._emb_dim

    def forward(self, observations) -> dict[str, th.Tensor]:
        player = [1, 0, 0, 0, 0, 0, 0, 0]
        opponent = [0, 1, 0, 0, 0, 0, 0, 0]
        card_in_hand = [0, 0, 1, 0, 0, 0, 0, 0]
        card_in_deck = [0, 0, 0, 1, 0, 0, 0, 0]
        friendly_creature_lane0 = [0, 0, 0, 0, 1, 0, 0, 0]
        friendly_creature_lane1 = [0, 0, 0, 0, 0, 1, 0, 0]
        opponent_creature_lane0 = [0, 0, 0, 0, 0, 0, 1, 0]
        opponent_creature_lane1 = [0, 0, 0, 0, 0, 0, 0, 1]

        # embedding of both players
        p = self.player_embedding(observations["player_stats"])
        op = self.player_embedding(observations["opponent_stats"])
        
        # add type and location information to the player embeddings
        p = th.cat((p, p.new_tensor(player).expand(p.size(0), -1)), dim=1)
        op = th.cat((op, op.new_tensor(opponent).expand(op.size(0), -1)), dim=1)
        
        # embedding of individual deck cards
        p_deck_cards = observations["player_deck"]
        p_deck_cards = self.card_embedding(p_deck_cards)
        
        # embedding of the whole deck
        p_deck = self.deck_embedding(p_deck_cards).squeeze(1)
        
        # add type and location information to the whole deck embedding
        p_deck = th.cat((
            p_deck,
            p_deck.new_tensor(card_in_deck).expand(p_deck.size(0), -1)
        ), dim=1)

        # embedding of individual hand cards
        p_hand_cards = observations["player_hand"]
        p_hand_cards = self.card_embedding(p_hand_cards)
        
        # add type and location information to the hand card embeddings
        p_hand_cards = th.cat((
            p_hand_cards, 
            p_hand_cards.new_tensor(card_in_hand).expand(p_hand_cards.size(0), p_hand_cards.size(1), -1)
        ), dim=2)
        
        # embedding of individual player lane 0 creatures
        p_lane0_creatures = observations["player_lane0"]
        p_lane0_creatures = self.card_embedding(p_lane0_creatures)
        
        # add type and location information to the lane creature embeddings
        p_lane0_creatures = th.cat((
            p_lane0_creatures, 
            p_lane0_creatures.new_tensor(friendly_creature_lane0).expand(p_lane0_creatures.size(0), p_lane0_creatures.size(1), -1)
        ), dim=2)
        
        # embedding of individual player lane 1 creatures
        p_lane1_creatures = observations["player_lane1"]
        p_lane1_creatures = self.card_embedding(p_lane1_creatures)

        # add type and location information to the lane creature embeddings
        p_lane1_creatures = th.cat((
            p_lane1_creatures, 
            p_lane1_creatures.new_tensor(friendly_creature_lane1).expand(p_lane1_creatures.size(0), p_lane1_creatures.size(1), -1)
        ), dim=2)
        
        # embedding of individual opponent lane 0 creatures
        op_lane0_creatures = observations["opponent_lane0"]
        op_lane0_creatures = self.card_embedding(op_lane0_creatures)
        
        # add type and location information to the lane creature embeddings
        op_lane0_creatures = th.cat((
            op_lane0_creatures, 
            op_lane0_creatures.new_tensor(opponent_creature_lane0).expand(op_lane0_creatures.size(0), op_lane0_creatures.size(1), -1)
        ), dim=2)

        # embedding of individual opponent lane 1 creatures
        op_lane1_creatures = observations["opponent_lane1"]
        op_lane1_creatures = self.card_embedding(op_lane1_creatures)
        
        # add type and location information to the lane creature embeddings
        op_lane1_creatures = th.cat((
            op_lane1_creatures, 
            op_lane1_creatures.new_tensor(opponent_creature_lane1).expand(op_lane1_creatures.size(0), op_lane1_creatures.size(1), -1)
        ), dim=2)

        # concat all and pass through a global embedding module
        state_input = th.cat((
            p.unsqueeze(1), op.unsqueeze(1), 
            p_deck.unsqueeze(1), p_hand_cards, 
            p_lane0_creatures, p_lane1_creatures, 
            op_lane0_creatures, op_lane1_creatures
        ), dim=1)  # [bs, 23, 40]
        
        all_tokens = self.global_embedding(state_input)  # [bs, 23, 64]

        # Slice to get entity groups
        hand_cards = all_tokens[:, 3:11, :]  # [bs, 8, 64]
        p_lane0_cr = all_tokens[:, 11:14, :]  # [bs, 3, 64]
        p_lane1_cr = all_tokens[:, 14:17, :]  # [bs, 3, 64]
        op_lane0_cr = all_tokens[:, 17:20, :]  # [bs, 3, 64]
        op_lane1_cr = all_tokens[:, 20:23, :]  # [bs, 3, 64]

        # Lane pooling
        p_lane0 = self.lane_pooling(p_lane0_cr).squeeze(1)  # [bs, 64]
        p_lane1 = self.lane_pooling(p_lane1_cr).squeeze(1)  # [bs, 64]

        # State pooling over all 23 tokens
        state = self.state_pooling(all_tokens).squeeze(1)  # [bs, 64]

        return {
            "hand_cards": hand_cards,
            "p_lane0_creatures": p_lane0_cr,
            "p_lane1_creatures": p_lane1_cr,
            "op_lane0_creatures": op_lane0_cr,
            "op_lane1_creatures": op_lane1_cr,
            "p_lane0": p_lane0,
            "p_lane1": p_lane1,
            "state": state,
            "action_mask": observations["action_mask"],
        }
