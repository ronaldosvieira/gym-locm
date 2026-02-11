import os
import urllib.request
from gym_locm.engine import State, Phase
from gym_locm.agents import Agent

from .agent import ByterlAgent

""" Implementation of ByteRL's agent for the Strategy Card Game AI Competition. Adapted from https://github.com/acatai/Strategy-Card-Game-AI-Competition. """

BYTERL_WEIGHTS_URL = "https://raw.githubusercontent.com/acatai/Strategy-Card-Game-AI-Competition/refs/heads/master/contest-2022-08-COG/ByteRL/Agent5.weights"


def ensure_weights_are_available():
    # calculate expected path of weights file
    weights_path = os.path.join(os.path.dirname(__file__), "Agent5.weights")

    # if the weights file does not exist, download it from the specified URL
    if not os.path.exists(weights_path):
        print(f"ByteRL's weights file not found at {weights_path}.")
        print(f"Downloading weights from {BYTERL_WEIGHTS_URL}...")

        urllib.request.urlretrieve(BYTERL_WEIGHTS_URL, weights_path)

        print("Download complete.")


class CardRecorder:
    def __init__(self) -> None:
        self.my_deck = []
        self.last_hands_instanceid = []

    def recorder_draft(self, chosen_card):
        self.my_deck.append(chosen_card)

    def recorder_battle(self, state):
        for card_hand in state.current_player.hand:
            if card_hand.instance_id not in self.last_hands_instanceid:
                for idx, card_deck in enumerate(self.my_deck):
                    if card_hand.id == card_deck.id:
                        self.my_deck.pop(idx)
                        break
        self.last_hands_instanceid = [
            card.instance_id for card in state.current_player.hand
        ]

    def full_missing_cards(self, state):
        if state.current_player.hand:
            instance_id = (
                max(card.instance_id for card in state.current_player.hand) + 1
            )
        else:
            instance_id = 0
        d1 = []
        assert len(self.my_deck) == len(state.current_player.deck), f"{len(self.my_deck)} != {len(state.current_player.deck)}"
        for card in self.my_deck:
            d1.append(card.make_copy(instance_id))
            instance_id += 1
        state.current_player.deck = d1
        return state


class ByteRL(Agent):
    def __init__(self):
        super().__init__()
        
        ensure_weights_are_available()

        self.reset()

    def seed(self, seed):
        pass

    def reset(self):
        self.agent = ByterlAgent()
        self.agent.reset()
        
        self.card_recorder = CardRecorder()
        
    def _simulate_deck_building(self, state):
        fake_state = State()
        
        self.card_recorder.my_deck = state.current_player.deck + state.current_player.hand
        
        for card in self.card_recorder.my_deck:
            fake_state.current_player.hand = [card] * 120
            
            _ = self.agent.act(fake_state)

    def act(self, state):
        if state.phase == Phase.DECK_BUILDING:
            action = self.agent.act(state)

            self.card_recorder.recorder_draft(state.current_player.hand[action.origin])

            return action

        elif state.phase == Phase.BATTLE:
            # if using the agent only in battle, simulate the deck-building phase
            if state.turn == 1 and self.card_recorder.my_deck == []:
                self._simulate_deck_building(state)
            
            self.card_recorder.recorder_battle(state)

            state = self.card_recorder.full_missing_cards(state.clone())

            return self.agent.act(state)
