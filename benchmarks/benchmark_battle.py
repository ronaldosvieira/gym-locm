"""
Benchmark script for gym-locm battle environment.

Measures:
  - Episodes/second (full game loop with random agents)
  - Steps/second
  - Reset time
  - Encode time
  - Optional cProfile output

Usage:
  python benchmarks/benchmark_battle.py                  # standard benchmark
  python benchmarks/benchmark_battle.py --profile        # with cProfile dump
  python benchmarks/benchmark_battle.py --episodes 1000  # custom episode count
"""

import argparse
import cProfile
import pstats
import time
from io import StringIO

import numpy as np


def run_benchmark(num_episodes=500, version="1.5", items=True, seed=42, verbose=True):
    """Run the battle environment benchmark and return timing results."""
    import gymnasium as gym
    from gym_locm.agents import RandomBattleAgent, RandomConstructedAgent, RandomDraftAgent

    # Choose deck-building agents based on version
    if version == "1.5":
        deck_agents = (RandomConstructedAgent(), RandomConstructedAgent())
    else:
        deck_agents = (RandomDraftAgent(), RandomDraftAgent())

    # Create env directly (bypass Gymnasium wrappers for accurate benchmarking)
    from gym_locm.envs.battle import LOCMBattleSingleEnv

    env = LOCMBattleSingleEnv(
        battle_agent=RandomBattleAgent(),
        deck_building_agents=deck_agents,
        seed=seed,
        version=version,
        items=items,
    )

    # Warmup (2 episodes)
    for _ in range(2):
        obs, info = env.reset()
        done = False
        while not done:
            # Pick a random valid action using action mask
            mask = env.action_masks()
            valid_actions = [i for i, m in enumerate(mask) if m]
            action = valid_actions[np.random.randint(len(valid_actions))] if valid_actions else 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    # Benchmark
    total_steps = 0
    reset_times = []
    step_times = []
    encode_times = []
    episode_times = []

    for ep in range(num_episodes):
        ep_start = time.perf_counter()

        # Time reset
        t0 = time.perf_counter()
        obs, info = env.reset()
        reset_times.append(time.perf_counter() - t0)

        done = False
        ep_steps = 0

        while not done:
            # Pick a random valid action using action mask
            mask = env.action_masks()
            valid_actions = [i for i, m in enumerate(mask) if m]
            action = valid_actions[np.random.randint(len(valid_actions))] if valid_actions else 0

            # Time step
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.perf_counter() - t0)

            # Time encode separately
            t0 = time.perf_counter()
            _ = env.encode_state()
            encode_times.append(time.perf_counter() - t0)

            done = terminated or truncated
            ep_steps += 1

        total_steps += ep_steps
        episode_times.append(time.perf_counter() - ep_start)

    env.close()

    # Compute statistics
    results = {
        "num_episodes": num_episodes,
        "total_steps": total_steps,
        "total_time": sum(episode_times),
        "episodes_per_sec": num_episodes / sum(episode_times),
        "steps_per_sec": total_steps / sum(episode_times),
        "avg_steps_per_episode": total_steps / num_episodes,
        "reset_mean_ms": np.mean(reset_times) * 1000,
        "reset_std_ms": np.std(reset_times) * 1000,
        "reset_p95_ms": np.percentile(reset_times, 95) * 1000,
        "step_mean_us": np.mean(step_times) * 1e6,
        "step_std_us": np.std(step_times) * 1e6,
        "step_p95_us": np.percentile(step_times, 95) * 1e6,
        "encode_mean_us": np.mean(encode_times) * 1e6,
        "encode_std_us": np.std(encode_times) * 1e6,
        "encode_p95_us": np.percentile(encode_times, 95) * 1e6,
    }

    if verbose:
        print_results(results, version, items)

    return results


def print_results(results, version, items):
    """Print a formatted benchmark results table."""
    print()
    print("=" * 60)
    print(f"  gym-locm Battle Benchmark (v{version}, items={items})")
    print("=" * 60)
    print()
    print(f"  Episodes:          {results['num_episodes']}")
    print(f"  Total steps:       {results['total_steps']}")
    print(f"  Total time:        {results['total_time']:.2f}s")
    print(f"  Avg steps/episode: {results['avg_steps_per_episode']:.1f}")
    print()
    print("  Throughput:")
    print(f"    Episodes/sec:    {results['episodes_per_sec']:.1f}")
    print(f"    Steps/sec:       {results['steps_per_sec']:.0f}")
    print()
    print("  Reset (per call):")
    print(f"    Mean:            {results['reset_mean_ms']:.2f} ms")
    print(f"    Std:             {results['reset_std_ms']:.2f} ms")
    print(f"    P95:             {results['reset_p95_ms']:.2f} ms")
    print()
    print("  Step (per call):")
    print(f"    Mean:            {results['step_mean_us']:.1f} µs")
    print(f"    Std:             {results['step_std_us']:.1f} µs")
    print(f"    P95:             {results['step_p95_us']:.1f} µs")
    print()
    print("  Encode (per call, extra call outside step):")
    print(f"    Mean:            {results['encode_mean_us']:.1f} µs")
    print(f"    Std:             {results['encode_std_us']:.1f} µs")
    print(f"    P95:             {results['encode_p95_us']:.1f} µs")
    print()
    print("=" * 60)


def run_profile(num_episodes=100, version="1.5", items=True, seed=42):
    """Run with cProfile and print top hotspots."""
    profiler = cProfile.Profile()
    profiler.enable()

    run_benchmark(num_episodes=num_episodes, version=version, items=items,
                  seed=seed, verbose=False)

    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    print(stream.getvalue())

    # Also print by tottime
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("tottime")
    stats.print_stats(40)
    print("\n--- Sorted by total time ---")
    print(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description="Benchmark gym-locm battle environment")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes to benchmark (default: 500)")
    parser.add_argument("--version", type=str, default="1.5",
                        choices=["1.2", "1.5"],
                        help="LOCM version (default: 1.5)")
    parser.add_argument("--no-items", action="store_true",
                        help="Disable items")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--profile", action="store_true",
                        help="Run with cProfile profiling")
    args = parser.parse_args()

    if args.profile:
        print(f"Profiling {args.episodes} episodes...")
        run_profile(
            num_episodes=args.episodes,
            version=args.version,
            items=not args.no_items,
            seed=args.seed,
        )
    else:
        run_benchmark(
            num_episodes=args.episodes,
            version=args.version,
            items=not args.no_items,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
