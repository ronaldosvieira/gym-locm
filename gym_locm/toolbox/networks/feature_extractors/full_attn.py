"""
Full cross-zone attention feature extractor — embeds all entities (players, deck,
hand cards, lane creatures, lane summaries) into a shared token sequence and applies
global Set Attention Blocks across all 30 tokens, with learnable additive encodings
for entity type, lane, and ownership.
"""
import torch as th
import torch.nn as nn
from gymnasium.spaces import Space

from gym_locm.toolbox.networks.feature_extractors.base import LOCMFeaturesExtractor
from gym_locm.toolbox.networks.attention import SAB, PMA


class FullAttnFeaturesExtractor(LOCMFeaturesExtractor):
    def __init__(
        self, 
        observation_space: Space, 
        card_dim: int = 17, 
        player_dim: int = 5, 
        creature_dim: int = 17,
        emb_dim: int = 64,
    ):
        super().__init__(
            observation_space, 
            features_dim=emb_dim
        )  # features_dim is used to calculate LSTM input size
        
        self._emb_dim = emb_dim

        self.player_embedding = nn.Sequential(
            nn.Linear(player_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
        )

        self.card_embedding = nn.Sequential(
            nn.Linear(card_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
        )
        
        self.creature_embedding = nn.Sequential(
            nn.Linear(creature_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
        )
        
        self.deck_embedding = nn.Sequential(
            SAB(emb_dim, emb_dim, num_heads=4, ln=True),
            PMA(dim=emb_dim, num_heads=4, num_seeds=4, ln=True),
        )
        
        self.lane_embedding = nn.Sequential(
            SAB(emb_dim, emb_dim, num_heads=4, ln=True),
            PMA(dim=emb_dim, num_heads=4, num_seeds=1, ln=True),
        )
        
        self.player_enc = nn.Parameter(th.randn(1, 1, emb_dim) * 0.02)
        self.opponent_enc = nn.Parameter(th.randn(1, 1, emb_dim) * 0.02)
        self.card_in_hand_enc = nn.Parameter(th.randn(1, 1, emb_dim) * 0.02)
        self.card_in_deck_enc = nn.Parameter(th.randn(1, 1, emb_dim) * 0.02)
        self.lane0_enc = nn.Parameter(th.randn(1, 1, emb_dim) * 0.02)
        self.lane1_enc = nn.Parameter(th.randn(1, 1, emb_dim) * 0.02)
        self.friendly_creature_enc = nn.Parameter(th.randn(1, 1, emb_dim) * 0.02)
        self.enemy_creature_enc = nn.Parameter(th.randn(1, 1, emb_dim) * 0.02)

        self.global_embedding = nn.Sequential(
            SAB(emb_dim, emb_dim, num_heads=4, ln=True),
            SAB(emb_dim, emb_dim, num_heads=4, ln=True),
        )
        
        self.state_embedding = nn.Sequential(
            PMA(dim=emb_dim, num_heads=4, num_seeds=1, ln=True),
        )

    @property
    def hand_cards_dim(self) -> int:
        return self._emb_dim

    @property
    def creature_tokens_dim(self) -> int:
        return self._emb_dim

    @property
    def lane_dim(self) -> int:
        return self._emb_dim

    @property
    def state_dim(self) -> int:
        return self._emb_dim

    def forward(self, observations) -> dict[str, th.Tensor]:
        # embedding of both players
        p = self.player_embedding(observations["player_stats"])  # [bs, emb_dim]
        op = self.player_embedding(observations["opponent_stats"])  # [bs, emb_dim]

        # add type and location information to the player embeddings
        p = p + self.player_enc.squeeze(1)  # [bs, emb_dim]
        op = op + self.opponent_enc.squeeze(1)  # [bs, emb_dim]

        # embedding of individual deck cards
        p_deck_cards = observations["player_deck"]
        p_deck_cards = self.card_embedding(p_deck_cards)

        # embedding of the whole deck
        p_deck = self.deck_embedding(p_deck_cards)  # [bs, 4, emb_dim]

        # add type and location information to the whole deck embedding
        p_deck = p_deck + self.card_in_deck_enc.expand(p_deck.size(0), p_deck.size(1), -1)  # [bs, 4, emb_dim]

        # embedding of individual hand cards
        p_hand_cards = observations["player_hand"]
        p_hand_cards = self.card_embedding(p_hand_cards)

        # add type and location information to the hand card embeddings
        p_hand_cards = p_hand_cards + self.card_in_hand_enc.expand(p_hand_cards.size(0), p_hand_cards.size(1), -1)

        # embedding of individual player lane 0 creatures
        p_lane0_creatures = observations["player_lane0"]
        p_lane0_creatures = self.creature_embedding(p_lane0_creatures)

        # add type and location information to the lane creature embeddings
        p_lane0_creatures = p_lane0_creatures + self.lane0_enc.expand(p_lane0_creatures.size(0), p_lane0_creatures.size(1), -1)
        p_lane0_creatures = p_lane0_creatures + self.friendly_creature_enc.expand(p_lane0_creatures.size(0), p_lane0_creatures.size(1), -1)

        # embedding of individual player lane 1 creatures
        p_lane1_creatures = observations["player_lane1"]
        p_lane1_creatures = self.creature_embedding(p_lane1_creatures)

        # add type and location information to the lane creature embeddings
        p_lane1_creatures = p_lane1_creatures + self.lane1_enc.expand(p_lane1_creatures.size(0), p_lane1_creatures.size(1), -1)
        p_lane1_creatures = p_lane1_creatures + self.friendly_creature_enc.expand(p_lane1_creatures.size(0), p_lane1_creatures.size(1), -1)

        # embedding of individual opponent lane 0 creatures
        op_lane0_creatures = observations["opponent_lane0"]
        op_lane0_creatures = self.creature_embedding(op_lane0_creatures)

        # add type and location information to the lane creature embeddings
        op_lane0_creatures = op_lane0_creatures + self.lane0_enc.expand(op_lane0_creatures.size(0), op_lane0_creatures.size(1), -1)
        op_lane0_creatures = op_lane0_creatures + self.enemy_creature_enc.expand(op_lane0_creatures.size(0), op_lane0_creatures.size(1), -1)

        # embedding of individual opponent lane 1 creatures
        op_lane1_creatures = observations["opponent_lane1"]
        op_lane1_creatures = self.creature_embedding(op_lane1_creatures)

        # add type and location information to the lane creature embeddings
        op_lane1_creatures = op_lane1_creatures + self.lane1_enc.expand(op_lane1_creatures.size(0), op_lane1_creatures.size(1), -1)
        op_lane1_creatures = op_lane1_creatures + self.enemy_creature_enc.expand(op_lane1_creatures.size(0), op_lane1_creatures.size(1), -1)
        
        p_lane0 = self.lane_embedding(p_lane0_creatures).squeeze(1)  # [bs, emb_dim]
        p_lane1 = self.lane_embedding(p_lane1_creatures).squeeze(1)  # [bs, emb_dim]
        op_lane0 = self.lane_embedding(op_lane0_creatures).squeeze(1)  # [bs, emb_dim]
        op_lane1 = self.lane_embedding(op_lane1_creatures).squeeze(1)  # [bs, emb_dim]

        # concat p, op, p_deck, p_hand_cards, p_lane0_creatures, p_lane1_creatures, op_lane0_creatures, op_lane1_creatures and pass through a global embedding module
        global_input = th.cat((
            p.unsqueeze(1), op.unsqueeze(1), 
            p_deck, p_hand_cards, 
            p_lane0.unsqueeze(1), p_lane1.unsqueeze(1), 
            op_lane0.unsqueeze(1), op_lane1.unsqueeze(1),
            p_lane0_creatures, p_lane1_creatures, 
            op_lane0_creatures, op_lane1_creatures
        ), dim=1)  # [bs, 1+1+4+8+1+1+1+1+3+3+3+3, emb_dim]
        
        all_entities = self.global_embedding(global_input)  # [bs, 30, emb_dim]
        state = self.state_embedding(all_entities).squeeze(1)  # [bs, emb_dim]

        return dict(
            hand_cards=all_entities[:, 6:14, :],  # [bs, 8, emb_dim]
            p_lane0_creatures=all_entities[:, 18:21, :],  # [bs, 3, emb_dim]
            p_lane1_creatures=all_entities[:, 21:24, :],
            op_lane0_creatures=all_entities[:, 24:27, :],
            op_lane1_creatures=all_entities[:, 27:30, :],
            p_lane0=all_entities[:, 14, :],  # [bs, emb_dim]
            p_lane1=all_entities[:, 15, :],  # [bs, emb_dim]
            state=state,  # [bs, emb_dim]
            action_mask=observations["action_mask"],
        )
