from gym_locm.toolbox.networks.feature_extractors.simple import SimpleFeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.deep_sets import DeepSetsFeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.set_transformer import SetTransformerFeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.set_transformer_2 import SetTransformer2FeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.set_transformer_2_1 import SetTransformer21FeaturesExtractor
from gym_locm.toolbox.networks.feature_extractors.gnn import GNNFeaturesExtractor

FEATURE_EXTRACTORS = {
    "simple": SimpleFeaturesExtractor,
    "deep_sets": DeepSetsFeaturesExtractor,
    "set_transformer": SetTransformerFeaturesExtractor,
    "set_transformer_2": SetTransformer2FeaturesExtractor,
    "set_transformer_2_1": SetTransformer21FeaturesExtractor,
    "gnn": GNNFeaturesExtractor,
}
