import sys
try:
    from os import fpathconf
except ImportError:
    fpathconf = None
from gym_locm.agents import Agent
from gym_locm.engine import Action, ActionType, Lane, State

try:
    import pexpect
except ImportError:
    pass


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class NativeAgent(Agent):
    action_buffer = []

    def __init__(self, cmd, stateful=True, verbose=False):
        self.cmd = cmd
        self.stateful = stateful
        self.verbose = verbose
        self.initialized = False
        self.action_buffer = []

        self.raw_actions = ""

        self._process = None

    def initialize(self):
        try:
            self._process: pexpect.pty_spawn.spawn = pexpect.spawn(
                self.cmd, echo=False, encoding="utf-8"
            )
        except NameError:
            raise ImportError(
                "To run native agents, please install `gym-locm[native-agent]`."
            )
        self.initialized = True

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.initialized:
            self._process.terminate()

            self._process = None
            self.initialized = False

    def seed(self, seed):
        pass

    def reset(self):
        self.action_buffer = []

        if self.initialized and self.stateful:
            self._process.terminate()

            self._process = None
            self.initialized = False

    @staticmethod
    def decode_actions(actions):
        actions = actions.split(";")
        decoded_actions = []

        for action in actions:
            tokens = action.split()

            if not tokens:
                continue

            if tokens[0] == "PASS":
                decoded_actions.append(Action(ActionType.PASS))
            elif tokens[0] == "PICK":
                decoded_actions.append(Action(ActionType.PICK, int(tokens[1])))
            elif tokens[0] == "CHOOSE":
                decoded_actions.append(Action(ActionType.CHOOSE, int(tokens[1])))
            elif tokens[0] == "USE":
                origin = int(tokens[1])
                target = int(tokens[2])

                if target == -1:
                    target = None

                decoded_actions.append(Action(ActionType.USE, origin, target))
            elif tokens[0] == "SUMMON":
                origin = int(tokens[1])
                target = Lane(int(tokens[2]))

                decoded_actions.append(Action(ActionType.SUMMON, origin, target))
            elif tokens[0] == "ATTACK":
                origin = int(tokens[1])
                target = int(tokens[2])

                if target == -1:
                    target = None

                decoded_actions.append(Action(ActionType.ATTACK, origin, target))

        return decoded_actions

    def act(self, state, multiple=False):
        if not self.initialized:
            self.initialize()

        return self._act(state, multiple)

    def _act(self, state, multiple=False):
        if self.action_buffer:
            if multiple:
                return list(reversed(self.action_buffer))
            else:
                return self.action_buffer.pop()

        # get max send buffer size
        if fpathconf is not None:
            n = fpathconf(0, "PC_MAX_CANON")
        else:
            n = 4096

        # get state as native string
        state_as_str = str(state)

        # separate state string in parts of up to n bytes each
        state_as_str_parts = [
            state_as_str[i : i + n] for i in range(0, len(state_as_str), n)
        ]

        bytes_sent = 0

        # send each part of the state to the agent
        for state_as_str_part in state_as_str_parts:
            bytes_sent += self._process.send(state_as_str_part)

            if self.verbose:
                eprint("Sent a total of", bytes_sent, "bytes")

        if self.verbose:
            print(
                "State bytes:",
                len(state_as_str.encode("utf-8")),
                "Bytes sent:",
                bytes_sent,
            )

        actions = []
        old_raw_output = ""

        try:
            i = 1
            raw_output = self._process.readline()

            while not actions and raw_output != old_raw_output:
                if self.verbose:
                    eprint(f"Trying to decode actions... (try {i})")

                # remove the \n
                self.raw_actions = raw_output.strip()

                if self.verbose:
                    eprint("Raw output:", self.raw_actions)

                actions = self.decode_actions(raw_output)

                if self.verbose:
                    eprint("Decoded:", actions)

                if not actions:
                    old_raw_output = raw_output
                    # read an action output ending with \n
                    raw_output = self._process.readline()

                i += 1

        except pexpect.TIMEOUT:
            print("WARNING: timeout")
        except pexpect.EOF:
            print("WARNING: eof")

        if not actions:
            raise Exception(
                "Failed to get actions from native agent. Last output line: "
                + old_raw_output
            )

        if actions[-1].type != ActionType.PASS and state.is_battle():
            actions += [Action(ActionType.PASS)]

        if multiple:
            return actions
        else:
            self.action_buffer = list(reversed(actions))

            return self.action_buffer.pop()

    def __repr__(self) -> str:
        return f"native:{self.cmd}"


class NativeBattleAgent(NativeAgent):
    def __init__(self, cmd, version="1.5", *args, **kwargs):
        super().__init__(cmd, *args, **kwargs)

        if version not in ("1.2", "1.5"):
            raise ValueError("version must be '1.2' or '1.5'")

        self.version = version

    def simulate_draft(self, state):
        fake_state = State(version=self.version)

        play_first = state.current_player.id == 0
        deck = state.current_player.deck + state.current_player.hand

        if not play_first:
            fake_state.act(Action(ActionType.PASS))

        for turn in range(state.deck_building_phase.n):
            chosen_card = deck[turn]

            fake_state.current_player.hand = [chosen_card] * state.deck_building_phase.k

            self._process.write(str(fake_state))

            try:
                raw_output = self._process.readline()

                if self.verbose:
                    eprint(raw_output, end="")
            except pexpect.TIMEOUT:
                print("WARNING: timeout")
            except pexpect.EOF:
                print("WARNING: eof")

        fake_state.act(Action(ActionType.PASS))

        if play_first:
            fake_state.act(Action(ActionType.PASS))

        try:
            raw_output = self._process.read_nonblocking(size=2048, timeout=0.1)

            if self.verbose:
                eprint(raw_output, end="")
        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            pass

    def simulate_constructed(self, state):
        # differently from simulate_draft, the agent here
        # is not forced to select the exact cards in their actual deck
        # most agents won't have a problem with this, but some might
        fake_state = State(version=self.version)

        self._process.write(str(fake_state))

        try:
            raw_output = self._process.readline()

            if self.verbose:
                eprint(raw_output, end="")
        except pexpect.TIMEOUT:
            print("WARNING: timeout")
        except pexpect.EOF:
            print("WARNING: eof")

        try:
            raw_output = self._process.read_nonblocking(size=2048, timeout=0.1)

            if self.verbose:
                eprint(raw_output, end="")
        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            pass

    def act(self, state, multiple=False):
        if not self.initialized:
            self.initialize()

            if self.version == "1.2":
                self.simulate_draft(state)
            elif self.version == "1.5":
                self.simulate_constructed(state)

        return super()._act(state, multiple)


class NativeDraftAgent(NativeAgent):
    pass


class NativeConstructedAgent(NativeAgent):
    pass
