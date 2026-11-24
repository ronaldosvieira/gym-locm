"""Actor-critic network registry for LOCM battle networks."""

import sys
import types

from gym_locm.toolbox.networks.actor_critics.simple import SimpleLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.enumerated import EnumeratedLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.typed import TypedLOCMNetwork
from gym_locm.toolbox.networks.actor_critics.unified import UnifiedLOCMNetwork

ACTOR_CRITICS = {
    "simple": SimpleLOCMNetwork,
    "enumerated": EnumeratedLOCMNetwork,
    "typed": TypedLOCMNetwork,
    "unified": UnifiedLOCMNetwork,
}

# Backward-compatibility alias for checkpoints saved before "autoreg"
# (AutoRegressiveLOCMNetwork in actor_critics/autoreg.py) was renamed to
# "unified" (UnifiedLOCMNetwork in actor_critics/unified.py). When SB3/cloudpickle
# unpickles those checkpoints, it imports `gym_locm.toolbox.networks.actor_critics.autoreg`
# and looks up `AutoRegressiveLOCMNetwork`, which resolves to `UnifiedLOCMNetwork`.
_autoreg_shim = types.ModuleType(__name__ + ".autoreg")
_autoreg_shim.AutoRegressiveLOCMNetwork = UnifiedLOCMNetwork
sys.modules[__name__ + ".autoreg"] = _autoreg_shim
autoreg = _autoreg_shim
