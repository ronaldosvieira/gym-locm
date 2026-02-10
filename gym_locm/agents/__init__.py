from typing import Type

from gym_locm.agents.base import (
    Agent,
    PassDraftAgent,
    PassConstructedAgent,
    PassBattleAgent,
    RandomDraftAgent,
    RandomConstructedAgent,
    RandomBattleAgent,
)
from gym_locm.agents.draft import (
    RuleBasedDraftAgent,
    MaxAttackDraftAgent,
    IceboxDraftAgent,
    ClosetAIDraftAgent,
    UJI1DraftAgent,
    UJI2DraftAgent,
    CoacDraftAgent,
    Coac2DraftAgent,
    ChadDraftAgent,
    HistorylessDraftAgent,
)
from gym_locm.agents.constructed import (
    MaxAttackConstructedAgent,
    InspiraiConstructedAgent,
)
from gym_locm.agents.battle import (
    GreedyBattleAgent,
    RuleBasedBattleAgent,
    MaxAttackBattleAgent,
)
from gym_locm.agents.native_agent import (
    NativeAgent,
    NativeBattleAgent,
    NativeDraftAgent,
    NativeConstructedAgent,
)
from gym_locm.agents.rl import RLDraftAgent, RLBattleAgent


draft_agents = {
    "pass": PassDraftAgent,
    "random": RandomDraftAgent,
    "rule-based": RuleBasedDraftAgent,
    "max-attack": MaxAttackDraftAgent,
    "baseline1": RuleBasedDraftAgent,
    "baseline2": MaxAttackDraftAgent,
    "icebox": IceboxDraftAgent,
    "closet-ai": ClosetAIDraftAgent,
    "uji1": UJI1DraftAgent,
    "uji2": UJI2DraftAgent,
    "coac": CoacDraftAgent,
    "coac2": Coac2DraftAgent,
    "chad": ChadDraftAgent,
    "historyless": HistorylessDraftAgent,
    "rl": RLDraftAgent,
}

constructed_agents = {
    "pass": PassConstructedAgent,
    "random": RandomConstructedAgent,
    "ma": MaxAttackConstructedAgent,
    "max-attack": MaxAttackConstructedAgent,
    "inspirai": InspiraiConstructedAgent,
}

battle_agents = {
    "pass": PassBattleAgent,
    "random": RandomBattleAgent,
    "greedy": GreedyBattleAgent,
    "osl": GreedyBattleAgent,
    "rule-based": RuleBasedBattleAgent,
    "max-attack": MaxAttackBattleAgent,
    "baseline1": RuleBasedBattleAgent,
    "baseline2": MaxAttackBattleAgent,
    "ma": MaxAttackBattleAgent,
    "rl": RLBattleAgent,
}


def parse_draft_agent(agent_name: str) -> Type:
    return draft_agents[agent_name.lower().replace(" ", "-")]


def parse_constructed_agent(agent_name: str) -> Type:
    return constructed_agents[agent_name.lower().replace(" ", "-")]


def parse_battle_agent(agent_name: str) -> Type:
    return battle_agents[agent_name.lower().replace(" ", "-")]
