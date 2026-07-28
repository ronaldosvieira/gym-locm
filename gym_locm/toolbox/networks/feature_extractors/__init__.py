"""Feature extractor registry for LOCM battle networks."""

from gym_locm.toolbox.networks.feature_extractors.simple import SimpleFeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.deep_sets import DeepSetsFeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.zone_attn import ZoneAttnFeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.full_attn import FullAttnFeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.gnn import GNNFeaturesExtractor

FEATURE_EXTRACTORS = {
    "simple": SimpleFeaturesExtractor,
    "deep_sets": DeepSetsFeaturesExtractor,
    "zone_attn": ZoneAttnFeaturesExtractor,
    "full_attn": FullAttnFeaturesExtractor,
    "gnn": GNNFeaturesExtractor,
}
