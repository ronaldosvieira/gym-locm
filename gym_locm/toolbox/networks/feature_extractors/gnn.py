"""
Graph Neural Network (GNN) feature extractor for LOCM.

Heterogeneous message-passing GNN that models the game state as a graph
of 25 nodes (players, deck, hand cards, lane creatures, lane summary nodes)
connected by typed edges (summon, use, attack, structural).
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from gym_locm.toolbox.networks.feature_extractors.base import LOCMFeaturesExtractor
from gym_locm.toolbox.networks.attention import MeanPool


class HeteroGNNLayer(nn.Module):
    """Single heterogeneous GNN message-passing layer with typed edges."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_types = ["summon", "use", "attack", "struct"]

        self.W = nn.ModuleDict({
            edge_type: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for edge_type in self.edge_types
        })
        self.W_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, X, A_dict):
        # X: [bs, 25, hidden_dim]
        # A_dict: dict of edge_type -> [bs, 25, 25] float tensor
        out = self.W_self(X)
        for edge_type in self.edge_types:
            A = A_dict[edge_type]
            # message passing: A @ X
            msg = th.bmm(A, X)
            out = out + self.W[edge_type](msg)

        out = X + F.relu(self.norm(out))  # residual connection
        return out


class GNNFeaturesExtractor(LOCMFeaturesExtractor):
    """Heterogeneous GNN feature extractor with action-mask-derived adjacencies."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__(observation_space, features_dim=features_dim, **kwargs)

        self.hidden_dim = features_dim
        self.num_layers = num_layers

        player_features = observation_space["player_stats"].shape[0]
        card_features = observation_space["player_hand"].shape[1]
        creature_features = observation_space["player_lane0"].shape[1]

        # Initial projections to project everything to hidden_dim
        self.player_proj = nn.Linear(player_features, self.hidden_dim)
        self.opponent_proj = nn.Linear(player_features, self.hidden_dim)
        self.card_proj = nn.Linear(card_features, self.hidden_dim)
        self.creature_proj = nn.Linear(creature_features, self.hidden_dim)

        # Learnable nodes for the two lanes
        self.lane0_node = nn.Parameter(th.randn(1, 1, self.hidden_dim))
        self.lane1_node = nn.Parameter(th.randn(1, 1, self.hidden_dim))

        self.gnn_layers = nn.ModuleList([
            HeteroGNNLayer(self.hidden_dim) for _ in range(self.num_layers)
        ])

        # Static structural adjacency (registered as buffer to avoid recomputation)
        A_struct = th.zeros(25, 25)
        A_struct[2:17, 0] = 1   # Deck, Hand, P_Lanes belong to Player
        A_struct[17:23, 1] = 1  # Op_Lanes belong to Opponent
        A_struct[11:14, 23] = 1 # P_Lane0 in Lane0
        A_struct[17:20, 23] = 1 # Op_Lane0 in Lane0
        A_struct[14:17, 24] = 1 # P_Lane1 in Lane1
        A_struct[20:23, 24] = 1 # Op_Lane1 in Lane1
        A_struct = A_struct + A_struct.t()
        self.register_buffer("A_struct_static", A_struct)

        self.state_pooling = MeanPool(dim=self.hidden_dim)

    @property
    def hand_cards_dim(self) -> int:
        return self.hidden_dim

    @property
    def creature_tokens_dim(self) -> int:
        return self.hidden_dim

    @property
    def lane_dim(self) -> int:
        return self.hidden_dim

    @property
    def state_dim(self) -> int:
        return self.hidden_dim

    def forward(self, observations) -> dict[str, th.Tensor]:
        bs = observations["player_stats"].size(0)
        device = observations["player_stats"].device

        # Project all raw features to hidden_dim
        p = self.player_proj(observations["player_stats"]).unsqueeze(1)  # [bs, 1, hidden]
        op = self.opponent_proj(observations["opponent_stats"]).unsqueeze(1) # [bs, 1, hidden]

        deck = self.card_proj(observations["player_deck"]).mean(dim=1, keepdim=True) # [bs, 1, hidden]
        hand = self.card_proj(observations["player_hand"]) # [bs, 8, hidden]

        p_lane0 = self.creature_proj(observations["player_lane0"]) # [bs, 3, hidden]
        p_lane1 = self.creature_proj(observations["player_lane1"]) # [bs, 3, hidden]
        op_lane0 = self.creature_proj(observations["opponent_lane0"]) # [bs, 3, hidden]
        op_lane1 = self.creature_proj(observations["opponent_lane1"]) # [bs, 3, hidden]

        lane0_emb = self.lane0_node.expand(bs, -1, -1) # [bs, 1, hidden]
        lane1_emb = self.lane1_node.expand(bs, -1, -1) # [bs, 1, hidden]

        # Stack into [bs, 25, hidden]
        X = th.cat([
            p, op, deck, hand, p_lane0, p_lane1, op_lane0, op_lane1, lane0_emb, lane1_emb
        ], dim=1)

        action_mask = observations["action_mask"].float()

        # Build Adjacency Matrices
        A_summon = th.zeros((bs, 25, 25), device=device)
        mask_summon = action_mask[:, 1:17].view(bs, 8, 2)
        A_summon[:, 3:11, 23:25] = mask_summon
        A_summon = A_summon + A_summon.transpose(1, 2)

        A_use = th.zeros((bs, 25, 25), device=device)
        mask_use = action_mask[:, 17:121].view(bs, 8, 13)
        A_use[:, 3:11, 1:2] = mask_use[:, :, 0:1] # Target opponent
        A_use[:, 3:11, 11:23] = mask_use[:, :, 1:13] # Target all creatures
        A_use = A_use + A_use.transpose(1, 2)

        A_attack = th.zeros((bs, 25, 25), device=device)
        mask_attack = action_mask[:, 121:145].view(bs, 6, 4)
        A_attack[:, 11:14, 1:2] = mask_attack[:, 0:3, 0:1]
        A_attack[:, 11:14, 17:20] = mask_attack[:, 0:3, 1:4]
        A_attack[:, 14:17, 1:2] = mask_attack[:, 3:6, 0:1]
        A_attack[:, 14:17, 20:23] = mask_attack[:, 3:6, 1:4]
        A_attack = A_attack + A_attack.transpose(1, 2)

        A_struct = self.A_struct_static.unsqueeze(0).expand(bs, -1, -1)

        A_dict = {
            "summon": A_summon,
            "use": A_use,
            "attack": A_attack,
            "struct": A_struct
        }

        # Message Passing
        for layer in self.gnn_layers:
            X = layer(X, A_dict)

        # Slicing the output tokens
        hand_cards = X[:, 3:11, :]
        p_lane0_cr = X[:, 11:14, :]
        p_lane1_cr = X[:, 14:17, :]
        op_lane0_cr = X[:, 17:20, :]
        op_lane1_cr = X[:, 20:23, :]
        p_lane0 = X[:, 23, :]
        p_lane1 = X[:, 24, :]

        # State pooling over all 25 tokens
        state = self.state_pooling(X).squeeze(1)

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
