import argparse
import cProfile
import io
import sys
from datetime import datetime
from pstats import Stats
from multiprocessing import Pool, Manager, Lock

from gym_locm import agents, engine

wins_by_p0 = Manager().list([0, 0])
lock = Lock()


def get_arg_parser():
    p = argparse.ArgumentParser(
        description="This is runner script for agent experimentation on gym-locm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    draft_choices = ["pass", "random", "rule-based", "max-attack",
                     "icebox", "closet-ai", "coac"]
    battle_choices = ["pass", "random", "greedy", "rule-based", "max-attack",
                      "coac", "mcts"]

    p.add_argument("--p1-draft", help="draft strategy used by player 1",
                   choices=draft_choices)
    p.add_argument("--p1-player", help="battle strategy used by player 1",
                   choices=battle_choices)
    p.add_argument("--p1-time", help="max thinking time for player 1",
                   default=200)
    p.add_argument("--p1-path",
                   help="native agent to be used by player 1 - "
                        "mutually exclusive with draft, battle and time args.")

    p.add_argument("--p2-draft", help="draft strategy used by player 2",
                   choices=draft_choices)
    p.add_argument("--p2-player", help="battle strategy used by player 2",
                   choices=battle_choices)
    p.add_argument("--p2-time", help="max thinking time for player 2",
                   default=200)
    p.add_argument("--p2-path",
                   help="native agent to be used by player 2 - "
                        "mutually exclusive with draft, battle and time args")

    p.add_argument("--games", type=int, help="amount of games to run",
                   default=1)
    p.add_argument("--processes", type=int, help="amount of processes to use",
                   default=1)
    p.add_argument("--seed", type=int, help="seed to use on episodes", default=0)
    p.add_argument("--profile", action="store_true",
                   help="whether to profile the runs (runs in a single process)")

    return p


def parse_agent(draft_agent, battle_agent):
    draft_choices = {
        "pass": agents.PassDraftAgent,
        "random": agents.RandomDraftAgent,
        "rule-based": agents.RuleBasedDraftAgent,
        "max-attack": agents.MaxAttackDraftAgent,
        "icebox": agents.IceboxDraftAgent,
        "closet-ai": agents.ClosetAIDraftAgent,
        "coac": agents.CoacDraftAgent
    }

    battle_choices = {
        "pass": agents.PassBattleAgent,
        "random": agents.RandomBattleAgent,
        "greedy": agents.GreedyBattleAgent,
        "rule-based": agents.RuleBasedBattleAgent,
        "max-attack": agents.MaxAttackBattleAgent,
        "coac": agents.CoacBattleAgent,
        "mcts": agents.MCTSBattleAgent
    }

    return draft_choices[draft_agent](), battle_choices[battle_agent]()


def evaluate(params):
    game_id, player_1, player_2, seed = params

    draft_bots = (player_1[0], player_2[0])
    battle_bots = (player_1[1], player_2[1])

    game = engine.Game(seed=seed + game_id)

    for bot in draft_bots + battle_bots:
        bot.reset()

    while game.winner is None:
        if game.phase == engine.Phase.DRAFT:
            bot = draft_bots[game.current_player.id]
        else:
            bot = battle_bots[game.current_player.id]

        action = bot.act(game)

        game.act(action)

    with lock:
        wins_by_p0[0] += 1 if game.winner == engine.PlayerOrder.FIRST else 0
        wins_by_p0[1] += 1

        wins, games = wins_by_p0

    ratio = 100 * wins / games

    print(f"{datetime.now()} Episode {games}: "
          f"{'%.2f' % ratio}% {'%.2f' % (100 - ratio)}%")

    return game.winner


def run():
    if sys.version_info < (3, 0, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if not args.p1_path and (not args.p1_draft or not args.p1_player):
        sys.stderr.write("You should use either p1-path or both "
                         "p1-draft and p1-player or p1-path.\n")
        sys.exit(1)
    elif not args.p2_path and (not args.p2_draft or not args.p2_player):
        sys.stderr.write("You should use either p2-path or both "
                         "p2-draft and p2-player.\n")
        sys.exit(1)

    if args.p1_path is not None:
        player_1 = agents.NativeAgent(args.p1_path)
        player_1 = (player_1, player_1)
    else:
        player_1 = parse_agent(args.p1_draft, args.p1_player)

    if args.p2_path is not None:
        player_2 = agents.NativeAgent(args.p2_path)
        player_2 = (player_2, player_2)
    else:
        player_2 = parse_agent(args.p2_draft, args.p2_player)

    if args.profile:
        profiler = cProfile.Profile()
        result = io.StringIO()

        profiler.enable()

        for i in range(args.games):
            evaluate((i, player_1, player_2, args.seed))

        profiler.disable()

        profiler_stats = Stats(profiler, stream=result)

        profiler_stats.sort_stats('cumulative')
        profiler_stats.print_stats()

        print(result.getvalue())
    else:
        params = ((j, player_1, player_2, args.seed) for j in range(args.games))

        with Pool(args.processes) as pool:
            pool.map(evaluate, params)


if __name__ == '__main__':
    run()
