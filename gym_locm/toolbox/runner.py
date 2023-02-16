import argparse
import cProfile
import io
import sys
from datetime import datetime
from pstats import Stats
from multiprocessing import Pool, Manager, Lock

from gym_locm import agents, engine
from gym_locm.agents import (
    parse_draft_agent,
    parse_battle_agent,
    parse_constructed_agent,
)

wins_by_p0 = Manager().list([0, 0])
lock = Lock()


def get_arg_parser():
    p = argparse.ArgumentParser(
        description="This is runner script for agent experimentation on gym-locm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--p1-deck-building",
        "-db1",
        help="deck-building agent used by player 1",
        choices=agents.draft_agents.keys(),
    )
    p.add_argument(
        "--p1-battle",
        "-b1",
        help="battle agent used by player 1",
        choices=agents.battle_agents.keys(),
    )
    p.add_argument("--p1-time", help="max thinking time for player 1", default=200)
    p.add_argument(
        "--p1-path",
        help="native agent to be used by player 1 - "
        "mutually exclusive with deck-building, battle and time args.",
    )

    p.add_argument(
        "--p2-deck-building",
        "-db2",
        help="deck-building agent used by player 2",
        choices=agents.draft_agents.keys(),
    )
    p.add_argument(
        "--p2-battle",
        "-b2",
        help="battle agent used by player 2",
        choices=agents.battle_agents.keys(),
    )
    p.add_argument("--p2-time", help="max thinking time for player 2", default=200)
    p.add_argument(
        "--p2-path",
        help="native agent to be used by player 2 - "
        "mutually exclusive with deck-building, battle and time args",
    )

    p.add_argument(
        "--version",
        "-v",
        type=str,
        choices=["1.5", "1.2"],
        default="1.5",
        help="version of LOCM to use; restricts list of agents",
    )
    p.add_argument("--games", type=int, help="amount of games to run", default=1)
    p.add_argument(
        "--processes", type=int, help="amount of processes to use", default=1
    )
    p.add_argument("--seed", type=int, help="seed to use on episodes", default=0)
    p.add_argument(
        "--silent", action="store_true", help="whether to print partial results"
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="whether to profile the runs (runs in a single process)",
    )
    p.add_argument(
        "--log-battles",
        action="store_true",
        help="whether to save a dataset of the battles run",
    )

    return p


def evaluate(params):
    game_id, player_1, player_2, seed, silent, log_battles, version = params

    deck_building_bots = (player_1[0], player_2[0])
    battle_bots = (player_1[1], player_2[1])

    game = engine.Game(seed=seed + game_id, version=version)

    for bot in deck_building_bots + battle_bots:
        bot.reset()

    battle_states = [], []

    while game.winner is None:
        if game.phase == engine.Phase.DECK_BUILDING:
            bot = deck_building_bots[game.current_player.id]
        else:
            if log_battles and not game.was_last_action_invalid:
                battle_states[game.current_player.id].append(str(game))

            bot = battle_bots[game.current_player.id]

        action = bot.act(game)

        game.act(action)

    with lock:
        wins_by_p0[0] += 1 if game.winner == engine.PlayerOrder.FIRST else 0
        wins_by_p0[1] += 1

        wins, games = wins_by_p0

        if log_battles:
            for player in list(engine.PlayerOrder):
                for battle_state in battle_states[player]:
                    print(1 if game.winner == player else 0)
                    print(battle_state)

    ratio = 100 * wins / games

    if not silent:
        print(
            f"{datetime.now()} Episode {games}: "
            f"{'%.2f' % ratio}% {'%.2f' % (100 - ratio)}%"
        )

    return game.winner


def run():
    if sys.version_info < (3, 0, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if not args.p1_path and (not args.p1_deck_building or not args.p1_battle):
        arg_parser.error(
            "You should use either p1-path or both p1-deck-building and p1-battle.\n"
        )
    elif not args.p2_path and (not args.p2_deck_building or not args.p2_battle):
        arg_parser.error(
            "You should use either p2-path or both p2-deck-building and p2-battle.\n"
        )

    if args.version == "1.5":
        parse_deck_building_agent = parse_constructed_agent
    else:
        parse_deck_building_agent = parse_draft_agent

    if args.p1_path is not None:
        player_1 = agents.NativeAgent(args.p1_path)
        player_1 = (player_1, player_1)
    else:
        player_1 = (
            parse_deck_building_agent(args.p1_deck_building)(),
            parse_battle_agent(args.p1_battle)(),
        )

    player_1[0].seed(args.seed)
    player_1[1].seed(args.seed)

    if args.p2_path is not None:
        player_2 = agents.NativeAgent(args.p2_path)
        player_2 = (player_2, player_2)
    else:
        player_2 = (
            parse_deck_building_agent(args.p2_deck_building)(),
            parse_battle_agent(args.p2_battle)(),
        )

    player_2[0].seed(args.seed)
    player_2[1].seed(args.seed)

    if args.profile:
        profiler = cProfile.Profile()
        result = io.StringIO()

        profiler.enable()

        for i in range(args.games):
            evaluate(
                (
                    i,
                    player_1,
                    player_2,
                    args.seed,
                    args.silent,
                    args.log_battles,
                    args.version,
                )
            )

        profiler.disable()

        profiler_stats = Stats(profiler, stream=result)

        profiler_stats.sort_stats("cumulative")
        profiler_stats.print_stats()

        print(result.getvalue())
    else:
        params = (
            (
                j,
                player_1,
                player_2,
                args.seed,
                args.silent,
                args.log_battles,
                args.version,
            )
            for j in range(args.games)
        )

        with Pool(args.processes) as pool:
            pool.map(evaluate, params)

    wins, games = wins_by_p0
    ratio = 100 * wins / games

    print(f"{'%.2f' % ratio}% {'%.2f' % (100 - ratio)}%")


if __name__ == "__main__":
    run()
