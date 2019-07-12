from gym_locm.engine import *


class BattleAgent:
    def act(self, state):
        pass


class PassBattleAgent(BattleAgent):
    def act(self, state):
        return BattleAction(BattleActionType.PASS)


class RuleBasedBattleAgent(BattleAgent):
    def __init__(self):
        self.last_action = None

    def act(self, state):
        current_player = state.players[state.current_player]
        opposing_player = state.players[state.current_player.opposing()]

        is_creature = lambda c: isinstance(c, Creature)
        is_green_item = lambda c: isinstance(c, GreenItem)
        is_red_item = lambda c: isinstance(c, RedItem)
        is_blue_item = lambda c: isinstance(c, BlueItem)
        has_enough_mana = lambda c: c.cost <= current_player.mana

        castable = list(filter(has_enough_mana, current_player.hand))
        summonable = list(filter(is_creature, castable))

        creatures = [c for lane in current_player.lanes for c in lane]
        opp_creatures = [c for lane in opposing_player.lanes for c in lane]
        can_attack = list(filter(Creature.able_to_attack, creatures))

        green_items = list(filter(is_green_item, castable))
        red_items = list(filter(is_red_item, castable))
        blue_items = list(filter(is_blue_item, castable))

        if summonable:
            creature = np.random.choice(summonable)
            lane = Lane.LEFT if np.random.choice([0, 1]) == 0 else Lane.RIGHT

            action = BattleAction(BattleActionType.SUMMON,
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

            action = BattleAction(BattleActionType.ATTACK,
                                  creature,
                                  np.random.choice(guards) if guards else None)

            if self.last_action != action:
                self.last_action = action
                return action

        if creatures and green_items:
            action = BattleAction(BattleActionType.USE,
                                  np.random.choice(green_items),
                                  np.random.choice(creatures))

            if self.last_action != action:
                self.last_action = action
                return action

        if opp_creatures and red_items:
            action = BattleAction(BattleActionType.USE,
                                  np.random.choice(red_items),
                                  np.random.choice(opp_creatures))

            if self.last_action != action:
                self.last_action = action
                return action

        if blue_items:
            action = BattleAction(BattleActionType.USE,
                                  np.random.choice(blue_items))

            if self.last_action != action:
                self.last_action = action
                return action

        action = BattleAction(BattleActionType.PASS)

        self.last_action = action

        return action
