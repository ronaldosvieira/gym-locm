import gymnasium as gym

from gym_locm import agents


def hello_world():
    p1 = agents.ByteRL()
    p2 = agents.ByteRL()

    env = gym.make(
        "LOCM-battle-2p-v0",
        deck_building_agents=(p1, p2),
        reward_functions=["win-loss", "opponent-health"],
        reward_weights=[1.0, 1.0],
        seed=42,
        render_mode="gui"
    )

    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        env.render()
        current_player = env.unwrapped.state.current_player.id
        agent = p1 if current_player == 0 else p2
        
        action = agent.act(env.unwrapped.state)
        print(f"Player {current_player} Action:", action)

        obs, reward, terminated, truncated, info = env.step(action)

        print("Reward:", reward, info["raw_rewards"])
        
    env.render()


if __name__ == "__main__":
    hello_world()
