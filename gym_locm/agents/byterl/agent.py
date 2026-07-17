import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from scipy.special import softmax
from gym_locm.agents import Agent
from gym_locm.engine import (
    Action,
    ActionType,
    Creature,
    GreenItem,
    RedItem,
    BlueItem,
    Lane,
    Phase,
    Player,
)
from .model import (
    fully_connected_layers,
    LSTM_layers,
    Conv1d,
    model_import,
    discrete_action_head,
    cb_discrete_action_head,
)

card_types = {Creature: 0, GreenItem: 1, RedItem: 2, BlueItem: 3}


def encode_card(card):
    """Encodes a card object into a numerical array."""
    card_type = [
        1.0 if isinstance(card, card_type) else 0.0 for card_type in card_types
    ]
    cost = card.cost / 12
    attack = card.attack / 12
    defense = max(-12, card.defense) / 12
    keywords = list(map(int, map(card.has_ability, "BCDGLW")))
    player_hp = card.player_hp / 12
    enemy_hp = card.enemy_hp / 12
    card_draw = card.card_draw / 2
    area = [1.0 if i == card.area else 0.0 for i in range(3)]

    return (
        card_type
        + [cost, attack, defense, player_hp, enemy_hp, card_draw]
        + keywords
        + area
    )


def encode_friendly_card_on_board(card: Creature):
    """Encodes a card object into a numerical array."""
    attack = card.attack / 12
    defense = card.defense / 12
    can_attack = int(card.can_attack and not card.has_attacked_this_turn)
    keywords = list(map(int, map(card.keywords.__contains__, "BCDGLW")))
    # no .area encoding is needed here

    return [attack, defense, can_attack] + keywords


def encode_enemy_card_on_board(card: Creature):
    """Encodes a card object into a numerical array."""
    attack = card.attack / 12
    defense = card.defense / 12
    keywords = list(map(int, map(card.keywords.__contains__, "BCDGLW")))
    # no .area encoding is needed here

    return [attack, defense] + keywords


def encode_current_player(current: Player):
    """encoding players, LOCM 1.5 no rune."""
    return current.health / 30, current.mana / 13, (1 + current.bonus_draw) / 6


def encode_opposing_player(opposing: Player):
    """encoding players, LOCM 1.5 no rune."""
    return (
        opposing.health / 30,
        (opposing.base_mana + opposing.bonus_mana) / 13,
        (1 + opposing.bonus_draw) / 6,
    )


def encode_players(current: Player, opposing: Player):
    """encoding players, LOCM 1.5 no rune."""
    return (
        current.health / 30,
        current.mana / 13,
        (1 + current.bonus_draw) / 6,
        opposing.health / 30,
        (opposing.base_mana + opposing.bonus_mana) / 13,
        (1 + opposing.bonus_draw) / 6,
    )


class SubmitInterface:
    def __init__(self):
        self.cards_num = 120
        self.selected_num = 30
        self.player_features = 3  # (hp, mana, next_draw). Note LOCM 1.5 has no rune
        self.cards_in_hand = 8
        self.card_features = 19  # make `.area` to be 3-dim in LOCM 1.5
        self.friendly_cards_on_board = 6
        self.friendly_board_card_features = 9
        self.enemy_cards_on_board = 6
        self.enemy_board_card_features = 8
        # 260 features if using items else TODO features
        self.state_size = (
            self.player_features * 2
            + self.cards_in_hand * self.card_features
            + self.friendly_cards_on_board * self.friendly_board_card_features
            + self.enemy_cards_on_board * self.enemy_board_card_features
        )
        self.now_phase = Phase.DECK_BUILDING
        self.card_set = None
        self.cards_selected_mask = None

    def env_to_agt_trans_observation(self, observation):
        """
        Args:
            observation: an LOCM `State` instance

        Returns:
            Transformed observations for each agent
        """
        state = observation  # shorthand

        p0, p1 = state.current_player, state.opposing_player
        
        # note: gym-locm addition - handle items=False
        items_setting = state.items
        state.items = True
        state._phase.items = True

        def fill_cards(card_list, up_to, features):
            remaining_cards = up_to - len(card_list)
            return card_list + [[0.0] * features for _ in range(remaining_cards)]

        self.now_phase = state.phase
        player = encode_current_player(p0)
        oppo_player = encode_opposing_player(p1)
        deck_cards = np.array(
            fill_cards(
                list(map(encode_card, p0.deck)), up_to=30, features=self.card_features
            ),
            dtype=np.float32,
        )
        # set obs None according to whose turn it is
        if self.now_phase == Phase.BATTLE:
            # get action mask, which has been made as for the current player (me player)
            action_mask = np.asarray(deepcopy(state.action_mask), np.float32)
            hand_cards = np.array(
                fill_cards(
                    list(map(encode_card, p0.hand)),
                    up_to=self.cards_in_hand,
                    features=self.card_features,
                ),
                dtype=np.float32,
            )
            me_lane0_cards = np.array(
                fill_cards(
                    list(map(encode_friendly_card_on_board, p0.lanes[0])),
                    up_to=3,
                    features=self.friendly_board_card_features,
                ),
                dtype=np.float32,
            )
            me_lane1_cards = np.array(
                fill_cards(
                    list(map(encode_friendly_card_on_board, p0.lanes[1])),
                    up_to=3,
                    features=self.friendly_board_card_features,
                ),
                dtype=np.float32,
            )
            oppo_lane0_cards = np.array(
                fill_cards(
                    list(map(encode_enemy_card_on_board, p1.lanes[0])),
                    up_to=3,
                    features=self.enemy_board_card_features,
                ),
                dtype=np.float32,
            )
            oppo_lane1_cards = np.array(
                fill_cards(
                    list(map(encode_enemy_card_on_board, p1.lanes[1])),
                    up_to=3,
                    features=self.enemy_board_card_features,
                ),
                dtype=np.float32,
            )

            if len(p0.deck) + len(p0.hand) == 30:  # find mask for last card
                self.cards_selected_mask = np.zeros((self.cards_num,), dtype=np.float32)
                for card in p0.hand + p0.deck:
                    self.cards_selected_mask[card.id] = 1
            cur_player_obs = OrderedDict(
                [
                    ("cb_phase_mask", np.array([0], dtype=np.float32)),
                    ("card_set", self.card_set),
                    ("cards_selected_mask", self.cards_selected_mask),
                    (
                        "cards_can_select_mask",
                        np.zeros((self.cards_num,), dtype=np.float32),
                    ),
                    ("hand_cards", hand_cards),
                    ("deck_cards", deck_cards),
                    ("me_lane0_cards", me_lane0_cards),
                    ("me_lane1_cards", me_lane1_cards),
                    ("oppo_lane0_cards", oppo_lane0_cards),
                    ("oppo_lane1_cards", oppo_lane1_cards),
                    ("player", player),
                    ("oppo_player", oppo_player),
                    ("bt_action_mask", action_mask),
                ]
            )
        else:
            cards_can_select_mask = np.asarray(deepcopy(state.action_mask), np.float32)
            action_mask = np.asarray(deepcopy([False] + list(state.action_mask)), np.float32)
            card_set_list = list(map(encode_card, p0.hand))
            self.card_set = np.array(card_set_list, dtype=np.float32)
            self.cards_selected_mask = np.zeros((self.cards_num,), dtype=np.float32)
            for card in p0.deck:
                self.cards_selected_mask[card.id] = 1
            cur_player_obs = OrderedDict(
                [
                    ("cb_phase_mask", np.array([1.0], dtype=np.float32)),
                    ("card_set", self.card_set),
                    ("cards_selected_mask", self.cards_selected_mask),
                    ("cards_can_select_mask", cards_can_select_mask),
                    ("deck_cards", deck_cards),
                    ("cb_action_mask", action_mask),
                    (
                        "hand_cards",
                        np.zeros(
                            (self.cards_in_hand, self.card_features), dtype=np.float32
                        ),
                    ),
                    ("deck_cards", deck_cards),
                    (
                        "me_lane0_cards",
                        np.zeros(
                            (3, self.friendly_board_card_features), dtype=np.float32
                        ),
                    ),
                    (
                        "me_lane1_cards",
                        np.zeros(
                            (3, self.friendly_board_card_features), dtype=np.float32
                        ),
                    ),
                    (
                        "oppo_lane0_cards",
                        np.zeros((3, self.enemy_board_card_features), dtype=np.float32),
                    ),
                    (
                        "oppo_lane1_cards",
                        np.zeros((3, self.enemy_board_card_features), dtype=np.float32),
                    ),
                    ("player", np.zeros(self.player_features, dtype=np.float32)),
                    ("oppo_player", np.zeros(self.player_features, dtype=np.float32)),
                ]
            )

        # note: gym-locm addition - handle items=False
        state.items = items_setting
        state._phase.items = items_setting

        return cur_player_obs

    def convert_to_locm_action(self, state, action_number):
        if state.phase == Phase.DECK_BUILDING:
            action_number -= 1
            return Action(ActionType.CHOOSE, action_number)

        me_player = state.current_player
        oppo_player = state.opposing_player

        if action_number == 0:
            return Action(ActionType.PASS)
        elif 1 <= action_number <= 16:
            action_number -= 1

            origin = int(action_number / 2)
            target = Lane(action_number % 2)

            origin = me_player.hand[origin].instance_id

            return Action(ActionType.SUMMON, origin, target)
        elif 17 <= action_number <= 120:
            action_number -= 17

            origin = int(action_number / 13)
            target = action_number % 13

            origin = me_player.hand[origin].instance_id

            if target == 0:
                target = None
            else:
                target -= 1

                side = [me_player, oppo_player][int(target / 6)]
                lane = int((target % 6) / 3)
                index = target % 3

                target = side.lanes[lane][index].instance_id

            return Action(ActionType.USE, origin, target)
        elif 121 <= action_number <= 144:
            action_number -= 121

            origin = action_number // 4
            target = action_number % 4

            lane = origin // 3

            origin = me_player.lanes[lane][origin % 3].instance_id

            if target == 0:
                target = None
            else:
                target -= 1

                target = oppo_player.lanes[lane][target].instance_id

            return Action(ActionType.ATTACK, origin, target)
        else:
            raise ValueError(f"Invalid action_number {action_number}")


class ByterlAgent(Agent):
    def __init__(self, temperature=0.0, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.np_random = np.random.RandomState(seed)

        weights_name = [
            "self/shared_raw_card_embed/conv0/weights:0",
            "self/shared_raw_card_embed/conv0/biases:0",
            "self/shared_raw_card_embed/conv1/weights:0",
            "self/shared_raw_card_embed/conv1/biases:0",
            "self/cb_mebed/att0/weights:0",
            "self/cb_mebed/att0/biases:0",
            "self/shared_me_lane_card_embed/weights:0",
            "self/shared_me_lane_card_embed/biases:0",
            "self/shared_oppo_lane_card_embed/weights:0",
            "self/shared_oppo_lane_card_embed/biases:0",
            "self/shared_player_embed/weights:0",
            "self/shared_player_embed/biases:0",
            "self/bt_fc0/weights:0",
            "self/bt_fc0/biases:0",
            "self/bt_fc1/weights:0",
            "self/bt_fc1/biases:0",
            "self/bt_fc2/weights:0",
            "self/bt_fc2/biases:0",
            "self/lstm_embed/rnn/multi_rnn_cell/cell_0/lstm_cell_layer_0/kernel:0",
            "self/lstm_embed/rnn/multi_rnn_cell/cell_0/lstm_cell_layer_0/bias:0",
            "self/lstm_embed/rnn/multi_rnn_cell/cell_0/lstm_cell_layer_0/projection/kernel:0",
            "self/cb_action_head/logits/weights:0",
            "self/cb_action_head/logits/biases:0",
            "self/bt_action_head/action/logits/weights:0",
            "self/bt_action_head/action/logits/biases:0",
            "self/vf/values/weights:0",
            "self/vf/values/biases:0",
        ]

        weights = model_import(
            os.path.join(os.path.dirname(__file__), "Agent5.weights"), weights_name
        )

        self.shared_raw_card_embed_cv0 = Conv1d(
            weights["self/shared_raw_card_embed/conv0/weights:0"],
            weights["self/shared_raw_card_embed/conv0/biases:0"],
        )
        self.shared_raw_card_embed_cv1 = Conv1d(
            weights["self/shared_raw_card_embed/conv1/weights:0"],
            weights["self/shared_raw_card_embed/conv1/biases:0"],
        )
        self.cb_embed_cv_att0 = Conv1d(
            weights["self/cb_mebed/att0/weights:0"],
            weights["self/cb_mebed/att0/biases:0"],
        )

        self.shared_me_lane_card_embed_cv = Conv1d(
            weights["self/shared_me_lane_card_embed/weights:0"],
            weights["self/shared_me_lane_card_embed/biases:0"],
        )
        self.shared_oppo_lane_card_embed_cv = Conv1d(
            weights["self/shared_oppo_lane_card_embed/weights:0"],
            weights["self/shared_oppo_lane_card_embed/biases:0"],
        )

        self.shared_player_embed_fc = fully_connected_layers(
            weights["self/shared_player_embed/weights:0"],
            weights["self/shared_player_embed/biases:0"],
        )

        self.bt_fc0 = fully_connected_layers(
            weights["self/bt_fc0/weights:0"], weights["self/bt_fc0/biases:0"]
        )
        self.bt_fc1 = fully_connected_layers(
            weights["self/bt_fc1/weights:0"], weights["self/bt_fc1/biases:0"]
        )
        self.bt_fc2 = fully_connected_layers(
            weights["self/bt_fc2/weights:0"], weights["self/bt_fc2/biases:0"]
        )

        self.bt_lstm_embed = LSTM_layers(
            weights[
                "self/lstm_embed/rnn/multi_rnn_cell/cell_0/lstm_cell_layer_0/kernel:0"
            ],
            weights[
                "self/lstm_embed/rnn/multi_rnn_cell/cell_0/lstm_cell_layer_0/bias:0"
            ],
            weights[
                "self/lstm_embed/rnn/multi_rnn_cell/cell_0/lstm_cell_layer_0/projection/kernel:0"
            ],
        )

        self.cb_action_head = cb_discrete_action_head(
            weights["self/cb_action_head/logits/weights:0"],
            weights["self/cb_action_head/logits/biases:0"],
        )

        self.bt_action_head = discrete_action_head(
            weights["self/bt_action_head/action/logits/weights:0"],
            weights["self/bt_action_head/action/logits/biases:0"],
        )
        self.cb_hs_new = None
        self.bt_hs_new = None
        self.intf = SubmitInterface()

    def _raw_card_feature_map(self, inputs):
        x = self.shared_raw_card_embed_cv0(inputs)
        outputs = self.shared_raw_card_embed_cv1(x)
        return outputs

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed)

    def reset(self, **kwargs):
        self.cb_hs_new = None
        self.bt_hs_new = None
        self.intf = SubmitInterface()

    def act(self, state):

        obs = self.intf.env_to_agt_trans_observation(state)

        card_set = np.expand_dims(obs["card_set"], axis=0)
        cards_fm = self._raw_card_feature_map(card_set)

        selected_mask = np.expand_dims(obs["cards_selected_mask"], axis=-1)
        selected_mask = np.tile(selected_mask, [1, 1, 32])

        sel_cards_fm = cards_fm * selected_mask
        sel_cards_emb = np.mean(sel_cards_fm, axis=1, keepdims=True)
        sel_cards_emb_to_return = np.squeeze(sel_cards_emb, axis=1)

        n_cards = obs["cards_selected_mask"].shape[0]
        sel_cards_emb = np.tile(sel_cards_emb, [1, n_cards, 1])
        x = np.concatenate([cards_fm, sel_cards_emb], axis=-1)
        x = self.cb_embed_cv_att0(x)

        can_sel_mask = np.expand_dims(obs["cards_can_select_mask"], axis=-1)
        can_sel_mask = np.tile(can_sel_mask, [1, 1, 64])
        cb_fm = x * can_sel_mask  # (bs, n_cards, 2*card_dim)
        selected_cards_embed = sel_cards_emb_to_return

        # (bs, 30, card_dim)
        deck_cards = np.expand_dims(obs["deck_cards"], axis=0)
        deck_cards_embed = self._raw_card_feature_map(deck_cards)
        deck_cards_embed = np.mean(deck_cards_embed, axis=1)  # (bs, card_dim)

        # (bs, 8, card_dim)
        hand_cards = np.expand_dims(obs["hand_cards"], axis=0)
        hand_cards_embed = self._raw_card_feature_map(hand_cards)
        hand_cards_embed = np.reshape(
            hand_cards_embed, shape=(1, -1)
        )  # (bs, 8*card_dim)

        # (bs, 3, dim)
        me_lane0_cards = np.expand_dims(obs["me_lane0_cards"], axis=0)
        me_lane0_cards_embed = self.shared_me_lane_card_embed_cv(me_lane0_cards)
        me_lane0_cards_embed = np.reshape(
            me_lane0_cards_embed, shape=(1, -1)
        )  # (bs, 3*dim)

        # (bs, 3, dim)
        me_lane1_cards = np.expand_dims(obs["me_lane1_cards"], axis=0)
        me_lane1_cards_embed = self.shared_me_lane_card_embed_cv(me_lane1_cards)
        me_lane1_cards_embed = np.reshape(
            me_lane1_cards_embed, shape=(1, -1)
        )  # (bs, 3*dim)

        # (bs, 3, dim)
        oppo_lane0_cards = np.expand_dims(obs["oppo_lane0_cards"], axis=0)
        oppo_lane0_cards_embed = self.shared_oppo_lane_card_embed_cv(oppo_lane0_cards)
        oppo_lane0_cards_embed = np.reshape(
            oppo_lane0_cards_embed, shape=(1, -1)
        )  # (bs, 3*dim)

        # (bs, 3, dim)
        oppo_lane1_cards = np.expand_dims(obs["oppo_lane1_cards"], axis=0)
        oppo_lane1_cards_embed = self.shared_oppo_lane_card_embed_cv(oppo_lane1_cards)
        oppo_lane1_cards_embed = np.reshape(
            oppo_lane1_cards_embed, shape=(1, -1)
        )  # (bs, 3*dim)

        # (bs, dim)
        player = np.expand_dims(obs["player"], axis=0)
        player_embed = self.shared_player_embed_fc(player)

        # (bs, dim)
        oppo_player = np.expand_dims(obs["oppo_player"], axis=0)
        oppo_player_embed = self.shared_player_embed_fc(oppo_player)

        #  # (bs, 10*card_dim+14*bt_base_dim)
        bt_embed = np.concatenate(
            [
                deck_cards_embed,
                hand_cards_embed,
                me_lane0_cards_embed,
                me_lane1_cards_embed,
                oppo_lane0_cards_embed,
                oppo_lane1_cards_embed,
                player_embed,
                oppo_player_embed,
            ],
            axis=-1,
        )

        bt_embed = np.concatenate([bt_embed, selected_cards_embed], axis=-1)

        bt_embed = self.bt_fc0(bt_embed)
        bt_embed = self.bt_fc1(bt_embed)
        bt_embed = self.bt_fc2(bt_embed)

        bt_lstm_embed, self.bt_hs_new = self.bt_lstm_embed(bt_embed, self.bt_hs_new)
        if state.phase == Phase.BATTLE:
            bt_action_logit = self.bt_action_head(bt_lstm_embed, obs["bt_action_mask"])
            
            if self.temperature == 0.0:
                bt_action_num = np.argmax(bt_action_logit, -1)
            else:
                action_prob = bt_action_logit / self.temperature
                action_prob = softmax(action_prob, -1)
                # action_prob = np.nan_to_num(action_prob, nan=0.0)

                bt_action_num = self.np_random.choice(len(action_prob), p=action_prob)

            return self.intf.convert_to_locm_action(state, bt_action_num)

        else:
            cb_action_logit = self.cb_action_head(
                cb_fm, obs["cb_action_mask"], obs["cards_can_select_mask"]
            )
            
            if self.temperature == 0.0:
                cb_action_num = np.argmax(cb_action_logit, -1)
            else:
                action_prob = cb_action_logit / self.temperature
                action_prob = softmax(action_prob, -1)
                # action_prob = np.nan_to_num(action_prob, nan=0.0)
                
                cb_action_num = self.np_random.choice(len(action_prob), p=action_prob)
            
            return self.intf.convert_to_locm_action(state, cb_action_num)
