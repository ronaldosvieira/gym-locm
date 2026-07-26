from gym_locm.toolbox.networks.actor_critics.simple import SimpleLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.type_specific import TypeSpecificLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.bilinear import BilinearLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.autoregressive import AutoregressiveLOCMNetwork

ACTOR_CRITICS = {
    "simple": SimpleLOCMNetwork,
    "type_specific": TypeSpecificLOCMNetwork,
    "bilinear": BilinearLOCMNetwork,
    "autoregressive": AutoregressiveLOCMNetwork,
}
