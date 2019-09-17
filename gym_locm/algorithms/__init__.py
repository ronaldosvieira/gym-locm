from abc import ABC, abstractmethod
from operator import attrgetter

import numpy as np

from gym_locm.algorithms.mcts import MCTS
from gym_locm.engine import PlayerOrder


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        """All possible successors of this board state"""
        return set()

    @abstractmethod
    def find_random_child(self):
        """Random successor of this board state (for more efficient simulation)"""
        return None

    @abstractmethod
    def is_terminal(self):
        """Returns True if the node has no children"""
        return True

    @abstractmethod
    def reward(self):
        """Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"""
        return 0

    @abstractmethod
    def __hash__(self):
        """Nodes must be hashable"""
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        """Nodes must be comparable"""
        return True


class LOCMNode(Node):
    def __init__(self, state, parent_node, action):
        self.state = state
        self.parent_node = parent_node
        self.action = action

    def find_children(self):
        """All possible successors of this board state"""
        actions = self.state.available_actions
        children = []

        for action in actions:
            state_copy = self.state.clone()
            state_copy.act(action)

            children.append(LOCMNode(state_copy, self, action))

        return set(children)

    def find_random_child(self):
        """Random successor of this board state (for more efficient simulation)"""
        state_copy = self.state.clone()

        action = np.random.choice(state_copy.available_actions)

        state_copy.act(action)

        return LOCMNode(state_copy, self, action)

    def is_terminal(self):
        """Returns True if the node has no children"""
        return not self.state.available_actions

    def reward(self):
        """Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"""
        return 1 if self.state.winner == PlayerOrder.FIRST else -1

    def __hash__(self):
        """Nodes must be hashable"""
        s = self.state
        p0, p1 = self.state.players

        attributes = [
            s.phase, s.turn, s.current_player.id,
            p0.health, p0.base_mana, p0.bonus_mana, p0.next_rune, p0.bonus_draw,
            p1.health, p1.base_mana, p1.bonus_mana, p1.next_rune, p1.bonus_draw,
            "p0", *[(c.id, c.instance_id) for c in sorted(p0.hand, key=attrgetter('id'))],
            "p1", *[(c.id, c.instance_id) for c in sorted(p1.hand, key=attrgetter('id'))]
        ]

        for p in (p0, p1):
            for j in range(2):
                for i in range(3):
                    if len(p.lanes[j]) > i:
                        c = sorted(p.lanes[j], key=attrgetter('id'))[i]

                        stats = [c.id, c.instance_id, c.attack, c.defense] + \
                                list(map(int, map(c.keywords.__contains__, 'BCDGLW'))) + \
                                [int(p.id), j, c.able_to_attack()]
                    else:
                        stats = [-1] * 12

                    attributes.extend(stats)

        return hash(tuple(attributes))

    def __eq__(self, other):
        """Nodes must be comparable"""
        return hash(self) == hash(other)
