"""Actor-critic network registry for LOCM battle networks."""

from gym_locm.toolbox.networks.actor_critics.simple import SimpleLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.typed import TypedLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.bilinear import BilinearLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.conditional import ConditionalLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.autoreg import AutoRegressiveLOCMNetwork

ACTOR_CRITICS = {
    "simple": SimpleLOCMNetwork,
    "typed": TypedLOCMNetwork,
    "bilinear": BilinearLOCMNetwork,
    "conditional": ConditionalLOCMNetwork,
    "autoreg": AutoRegressiveLOCMNetwork,
}
