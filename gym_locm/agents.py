from abc import ABC, abstractmethod

from gym_locm.engine import *
from gym_locm.helpers import *


class Agent(ABC):
    @abstractmethod
    def act(self, state):
        pass


class PassBattleAgent(Agent):
    def act(self, state):
        return Action(ActionType.PASS)


class RandomBattleAgent(Agent):
    def act(self, state):
        return np.random.choice(state.available_actions)


class RuleBasedBattleAgent(Agent):
    def __init__(self):
        self.last_action = None

    def act(self, state):
        current_player = state.players[state.current_player]
        opposing_player = state.players[state.current_player.opposing()]

        castable = list(filter(has_enough_mana(current_player.mana),
                               current_player.hand))
        summonable = list(filter(is_it(Creature), castable))

        creatures = [c for lane in current_player.lanes for c in lane]
        opp_creatures = [c for lane in opposing_player.lanes for c in lane]
        can_attack = list(filter(Creature.able_to_attack, creatures))

        green_items = list(filter(is_it(GreenItem), castable))
        red_items = list(filter(is_it(RedItem), castable))
        blue_items = list(filter(is_it(BlueItem), castable))

        if summonable:
            creature = np.random.choice(summonable)
            lane = Lane.LEFT if np.random.choice([0, 1]) == 0 else Lane.RIGHT

            action = Action(ActionType.SUMMON,
                            creature,
                            lane)

            if self.last_action != action:
                self.last_action = action
                return action

        if can_attack:
            creature = np.random.choice(can_attack)
            lane = Lane.LEFT if creature in current_player.lanes[0] else Lane.RIGHT

            opp_creatures = opposing_player.lanes[lane]
            guards = list(filter(lambda c: c.has_ability('G'), opp_creatures))

            action = Action(ActionType.ATTACK,
                            creature,
                            np.random.choice(guards) if guards else None)

            if self.last_action != action:
                self.last_action = action
                return action

        if creatures and green_items:
            action = Action(ActionType.USE,
                            np.random.choice(green_items),
                            np.random.choice(creatures))

            if self.last_action != action:
                self.last_action = action
                return action

        if opp_creatures and red_items:
            action = Action(ActionType.USE,
                            np.random.choice(red_items),
                            np.random.choice(opp_creatures))

            if self.last_action != action:
                self.last_action = action
                return action

        if blue_items:
            action = Action(ActionType.USE,
                            np.random.choice(blue_items))

            if self.last_action != action:
                self.last_action = action
                return action

        action = Action(ActionType.PASS)

        self.last_action = action

        return action


PassDraftAgent = PassBattleAgent
RandomDraftAgent = RandomBattleAgent
