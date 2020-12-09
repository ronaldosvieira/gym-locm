import numpy as np

from gym_locm.agents import MaxAttackDraftAgent, MaxAttackBattleAgent
from gym_locm.envs.draft import LOCMDraftSingleTabularEnv, LOCMDraftEnv


def _new_q_table(env: LOCMDraftEnv):
    q = dict()

    for c1 in range(160):
        for c2 in range(c1 + 1, 160):
            for c3 in range(c2 + 1, 160):
                for action in range(env.k):
                    q[c1, c2, c3, action] = 0

    q[None] = 0

    return q


def _get_policy_for_state(q: dict, state: tuple):
    return np.argmax([q[(*state, 0)], q[(*state, 1)], q[(*state, 2)]])


def run():
    env = LOCMDraftSingleTabularEnv(
        draft_agent=MaxAttackDraftAgent(),
        battle_agents=(MaxAttackBattleAgent(), MaxAttackBattleAgent()),
        evaluation_battles=1
    )

    n_episodes = 1_000_000
    gamma = 1
    alpha = lambda i: 0.5 / i if i > 0 else 0
    epsilon = lambda i: 1.0 - (i / n_episodes)

    q = _new_q_table(env)

    for iteration in range(n_episodes * 30):
        if (iteration + 1) % 1000 == 0:
            print(f"Iteration {iteration + 1}")

        state = env.reset()
        done = False

        while not done:
            best_action = _get_policy_for_state(q, state)

            if np.random.random() <= epsilon(iteration):
                action = env.action_space.sample()
            else:
                action = best_action

            new_state, reward, done, info = env.step(action)

            if new_state is not None:
                best_new_action = _get_policy_for_state(q, new_state)
            else:
                best_new_action = 0

            q[(*state, action)] = q[(*state, action)] \
                + alpha(iteration) \
                * (reward + gamma * best_new_action - q[(*state, action)])

            state = new_state

    with open('policy.csv', 'w+') as policy:
        policy.write("c1;c2;c3;pi\n")

        for c1 in range(160):
            for c2 in range(c1 + 1, 160):
                for c3 in range(c2 + 1, 160):
                    policy.write(f"{c1};{c2};{c3};{_get_policy_for_state(q, (c1, c2, c3))}")
                    policy.write("\n")

    print("✅")


if __name__ == '__main__':
    run()
