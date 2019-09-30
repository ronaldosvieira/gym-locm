"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

Modified by Ronaldo Vieira, 2019
Original version:
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
import random
from collections import defaultdict
import math
from operator import attrgetter

import numpy as np

from gym_locm.engine import PlayerOrder


class Node:
    def __init__(self, state, actions, parent):
        self.state = state
        self.actions = actions
        self.parent = parent
        self.player_id = parent.player_id if parent else state.current_player.id

        self._hash = None

    def __hash__(self):
        """Nodes must be hashable"""
        if self._hash is not None:
            return self._hash

        s = self.state
        p0, p1 = self.state.players
        cp = self.state.players[self.player_id]

        attributes = [
            s.phase, s.turn, s.current_player.id,
            p0.health, p0.base_mana + p0.bonus_mana, p0.bonus_draw,
            p1.health, p1.base_mana + p1.bonus_mana, p1.bonus_draw
        ]

        attributes.extend(c.instance_id
                          for c in sorted(cp.hand, key=attrgetter('id')))

        for p in (p0, p1):
            for j in range(2):
                for i in range(3):
                    if len(p.lanes[j]) > i:
                        c = sorted(p.lanes[j], key=attrgetter('id'))[i]

                        stats = [c.instance_id, c.attack, c.defense] + \
                            list(map(int, map(c.keywords.__contains__, 'BCDGLW'))) + \
                            [int(p.id), j, c.able_to_attack()]
                    else:
                        stats = [-1] * 12

                    attributes.extend(stats)

        for action in self.actions:
            attributes.extend((action.type, action.origin, action.target))

        self._hash = hash(tuple(attributes))

        return self._hash

    def __eq__(self, other):
        """Nodes must be comparable"""
        return hash(self) == hash(other)


class MCTS:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, agents, exploration_weight=1.41):
        self.agents = agents
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = defaultdict(list)  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, state):
        node = Node(state, [], None)

        """Choose the best successor of node. (Choose a move in the game)"""
        if state.winner is not None:
            raise RuntimeError("choose called on terminal node")

        if node not in self.children:
            index = int(len(state.available_actions) * random.random())

            return state.available_actions[index]

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score).actions[-1]

    def do_rollout(self, state):
        node = Node(state, [], None)

        """"Make the tree one layer better. (Train for one iteration.)"""
        path, new_state = self._select(node)
        leaf = path[-1]

        self._expand(leaf, state, new_state)

        reward = self._simulate(new_state)
        self._backpropagate(path, reward)

    def _select(self, node):
        """Find an unexplored descendent of `node`"""
        path = []

        state_copy = node.state.clone()

        while True:
            path.append(node)

            if node not in self.children or not self.children[node] \
                    or state_copy.winner is not None:
                # node is either unexplored or terminal
                return path, state_copy

            unexplored = [item for item in self.children[node]
                          if item not in self.children.keys()]

            if unexplored:
                n = unexplored.pop()
                state_copy.act(n.actions[-1])
                path.append(n)
                return path, state_copy

            node = self._uct_select(node)  # descend a layer deeper
            state_copy.act(node.actions[-1])

    def _expand(self, node, root_state, new_state):
        """"Update the `children` dict with the children of `node`"""
        if node in self.children:
            return  # already expanded

        children = []

        for action in new_state.available_actions:
            children.append(Node(root_state, node.actions + [action], node))

        self.children[node] = children

    def _simulate(self, new_state):
        """Returns the reward for a random simulation (to completion) of `node`"""

        amount_deck = len(new_state.opposing_player.deck)
        amount_hand = len(new_state.opposing_player.hand)

        new_deck = []

        for i in range(amount_deck + amount_hand):
            random_index = int(3 * random.random())

            card = new_state._draft_cards[i][random_index].make_copy()

            new_deck.append(card)

        random.shuffle(new_deck)

        new_state.opposing_player.deck = new_deck
        new_state.opposing_player.hand = []

        new_state.opposing_player.draw(amount=amount_hand)

        for player in new_state.players:
            random.shuffle(player.deck)

        while new_state.winner is None:
            action = self.agents[new_state.current_player.id].act(new_state)
            new_state.act(action)

        return 1 if new_state.winner == PlayerOrder.FIRST else -1

    def _backpropagate(self, path, reward):
        """Send the reward back up to the ancestors of the leaf"""
        for node in reversed(path):
            self.N[node] += 1

            if node.state.current_player.id == PlayerOrder.FIRST:
                self.Q[node] += reward
            else:
                self.Q[node] -= reward

    def _uct_select(self, node):
        """Select a child of node, balancing exploration & exploitation"""

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_n_vertex = math.log(self.N[node])

        def uct(n):
            """Upper confidence bound for trees"""
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_n_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
