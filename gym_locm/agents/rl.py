from gym_locm.agents import Agent


class RLDraftAgent(Agent):
    def __init__(self, model):
        self.model = model

        self.hidden_states = None
        self.dones = None

    def seed(self, seed):
        pass

    def reset(self):
        self.hidden_states = None
        self.dones = None

    def act(self, state):
        action, self.hidden_states = self.model.predict(
            state,
            state=self.hidden_states,
            episode_start=self.dones,
            deterministic=True,
        )

        return action


class RLBattleAgent(Agent):
    def __init__(self, model, deterministic=False):
        self.model = model
        self.deterministic = deterministic

        self.hidden_states = None
        self.dones = None

    def seed(self, seed):
        pass

    def reset(self):
        self.hidden_states = None
        self.dones = None

    def act(self, state):
        action, self.hidden_states = self.model.predict(
            state,
            state=self.hidden_states,
            deterministic=self.deterministic,
        )

        return action
