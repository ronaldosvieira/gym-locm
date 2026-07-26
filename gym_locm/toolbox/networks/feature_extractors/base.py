"""
Base class for LOCM feature extractors.

All feature extractors (except Simple) must output a standardized dictionary:
{
    "hand_cards":         [bs, 8, card_emb_dim],
    "p_lane0_creatures":  [bs, 3, creature_emb_dim],
    "p_lane1_creatures":  [bs, 3, creature_emb_dim],
    "op_lane0_creatures": [bs, 3, creature_emb_dim],
    "op_lane1_creatures": [bs, 3, creature_emb_dim],
    "p_lane0":            [bs, lane_emb_dim],
    "p_lane1":            [bs, lane_emb_dim],
    "state":              [bs, state_emb_dim],
    "action_mask":        [bs, 145],
}

The Simple extractor outputs {"latent": [bs, feat_dim], "action_mask": [bs, 145]}
and can only be paired with SimpleLOCMNetwork.
"""

from abc import abstractmethod
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LOCMFeaturesExtractor(BaseFeaturesExtractor):
    """Base class for LOCM feature extractors that output entity-level embeddings."""

    @property
    @abstractmethod
    def hand_cards_dim(self) -> int:
        """Dimension of hand card embeddings."""
        ...

    @property
    @abstractmethod
    def creature_tokens_dim(self) -> int:
        """Dimension of lane creature embeddings."""
        ...

    @property
    @abstractmethod
    def lane_dim(self) -> int:
        """Dimension of pooled lane embeddings."""
        ...

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the global state embedding."""
        ...
