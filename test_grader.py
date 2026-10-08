"""Run agent.py on the extra maps in test_maps/ and show the score of each map (README 9.2)."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from agent import Agent
from environment import ACTIONS, MapDefinition
from public_grader import evaluate


def optimal_steps(definition):
    """Shortest path from start to goal on the full map (BFS)."""
    queue = deque([(definition.start, 0)])
    seen = {definition.start}
    while queue:
        (row, col), dist = queue.popleft()
        if (row, col) == definition.goal:
            return dist
        for dr, dc in ACTIONS.values():
            nxt = (row + dr, col + dc)
            if (0 <= nxt[0] < definition.rows and 0 <= nxt[1] < definition.cols
                    and definition.grid[nxt[0]][nxt[1]] != "#" and nxt not in seen):
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def score(result, optimal):
    if not result["success"]:
        return 0.0
    movement_steps = result["steps"] - result["collisions"]
    efficiency_ratio = min(1, optimal / movement_steps)
    return round(80 + 20 * efficiency_ratio - min(10, result["collisions"]), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    maps = sorted((Path(__file__).parent / "test_maps").glob("*.json"))
    agent = Agent()
    header = f"{'Map':16} {'Size':6} {'Style':12} {'Result':8} {'Steps':>6} {'Optimal':>8} {'Collisions':>11} {'Score':>7}"
    print("\n" + header)
    print("-" * len(header))

    scores = []
    for path in maps:
        definition = MapDefinition.from_json(path)
        optimal = optimal_steps(definition)
        try:
            result = evaluate(agent, path, args.verbose)
            status = "SUCCESS" if result["success"] else "FAILED"
            map_score = score(result, optimal)
            steps, collisions = result["steps"], result["collisions"]
        except Exception as exc:
            status, map_score, steps, collisions = "ERROR", 0.0, "-", "-"
            print(f"  Error on {path.stem}: {exc}")
        scores.append(map_score)
        size = f"{definition.rows}x{definition.cols}"
        print(f"{definition.name:16} {size:6} {definition.style:12} {status:8} {steps:>6} {optimal:>8} {collisions:>11} {map_score:>7.2f}")

    print("-" * len(header))
    print(f"Successful maps : {sum(s > 0 for s in scores)}/{len(scores)}")
    print(f"Average score   : {sum(scores) / len(scores):.2f}\n")


if __name__ == "__main__":
    main()
