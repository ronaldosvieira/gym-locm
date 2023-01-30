import sys
import argparse
import threading

from datetime import datetime
from gym_locm import agents, engine


def cmdline_args():
    p = argparse.ArgumentParser(
        description="This is the tester script for native agent experimentation on gym-locm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--p1", help="agent used by player 1", required=True)
    p.add_argument("--p2", help="agent used by player 2", required=True)
    p.add_argument("--games", type=int, help="amount of games to run", default=1)
    p.add_argument("--threads", type=int, help="amount of threads to use", default=1)

    return p.parse_args()


def evaluate(player_1, player_2):
    global i, wins

    bots = (agents.NativeAgent(player_1), agents.NativeAgent(player_2))

    while i < args.games:
        i += 1
        current_episode = i

        game = engine.Game(seed=i, version="1.2")

        for bot in bots:
            bot.reset()

        while game.winner is None:
            bot = bots[game.current_player.id]

            action = bot.act(game)

            game.act(action)

        if game.winner == engine.PlayerOrder.FIRST:
            wins += 1

        ratio = round(100 * wins / i, 2)

        print(
            f"{datetime.now()} Episode {current_episode}: {'%.2f' % ratio}% {'%.2f' % (100 - ratio)}%"
        )


if __name__ == "__main__":

    if sys.version_info < (3, 0, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    args = cmdline_args()

    i = 0
    wins = 0

    threads = []

    for _ in range(args.threads):
        thread = threading.Thread(target=evaluate, args=(args.p1, args.p2), daemon=True)
        thread.start()

        threads.append(thread)

    for thread in threads:
        thread.join()
