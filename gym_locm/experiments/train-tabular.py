import multiprocessing

import numpy as np

from gym_locm.agents import MaxAttackDraftAgent, MaxAttackBattleAgent
from gym_locm.envs.draft import LOCMDraftSingleTabularEnv


def _init_q_table():
    q_table = dict()

    for c1 in range(160):
        for c2 in range(c1 + 1, 160):
            for c3 in range(c2 + 1, 160):
                for action in range(k):
                    q_table[c1, c2, c3, action] = 0

    q_table[None] = 0

    return q_table


def _get_policy_for_state(state: tuple):
    try:
        return np.argmax([q[(*state, 0)], q[(*state, 1)], q[(*state, 2)]])
    except TypeError:
        return 0


def _get_best_q_for_state(state: tuple):
    try:
        return np.max([q[(*state, 0)], q[(*state, 1)], q[(*state, 2)]])
    except TypeError:
        return 0


def _iterate():
    with iteration.get_lock():
        iteration.value += 1

        return iteration.value


def q_learning(worker_id: int):
    env = LOCMDraftSingleTabularEnv(
        draft_agent=MaxAttackDraftAgent(),
        battle_agents=(MaxAttackBattleAgent(), MaxAttackBattleAgent()),
        evaluation_battles=1, k=k, n=n
    )

    state = env.reset()

    print(f"Worker {worker_id} starting")

    while True:
        current_iteration = _iterate()

        if current_iteration >= n_episodes * n:
            break

        if (current_iteration + 1) % 1000 == 0:
            print(f"Iteration {current_iteration + 1}")

        best_action = _get_policy_for_state(state)

        if np.random.random() <= epsilon(current_iteration):
            action = env.action_space.sample()
        else:
            action = best_action

        new_state, reward, done, info = env.step(action)

        best_q_new_state = _get_best_q_for_state(new_state)

        q[(*state, action)] += \
            alpha(current_iteration) \
            * (reward + gamma * best_q_new_state - q[(*state, action)])

        if q[(*state, action)] != 0:
            with test.get_lock():
                test.value += 1

        state = new_state if not done else env.reset()


def run():
    processes = multiprocessing.Pool(processes=4)

    for i in range(4):
        processes.apply_async(q_learning, [i])

    processes.close()
    processes.join()

    print(f"non-null updates: {test.value}")

    print("Saving policy...")

    with open('policy.csv', 'a') as policy:
        policy.write("c1;c2;c3;pi\n")

        for c1 in range(160):
            for c2 in range(c1 + 1, 160):
                for c3 in range(c2 + 1, 160):
                    policy.write(f"{c1};{c2};{c3};{_get_policy_for_state((c1, c2, c3))}\n")

    print("✅")


if __name__ == '__main__':
    k = 3
    n = 30

    n_episodes = 1_000_000
    gamma = 1
    alpha = lambda i: 0.5 / i if i > 0 else 0
    epsilon = lambda i: 1.0 - (i / (n_episodes * n))

    manager = multiprocessing.Manager()
    q = manager.dict(_init_q_table())
    iteration = multiprocessing.Value('l', 0)
    test = multiprocessing.Value('l', 0)

    run()
