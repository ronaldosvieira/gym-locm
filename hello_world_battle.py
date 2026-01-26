import gymnasium as gym

from gym_locm import agents


def hello_world():
    env = gym.make(
        "LOCM-battle-v0",
        deck_building_agents=(
            agents.RandomConstructedAgent(),
            agents.RandomConstructedAgent(),
        ),
        battle_agent=(agents.RandomBattleAgent()),
        reward_functions=["win-loss", "opponent-health"],
        reward_weights=[1.0, 1.0],
        seed=42,
    )

    agent = agents.GreedyBattleAgent()

    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        env.render()
        action = agent.act(env.state)
        print("Action:", action)

        obs, reward, terminated, truncated, info = env.step(action)

        print("Reward:", reward, info["raw_rewards"])


if __name__ == "__main__":
    hello_world()
