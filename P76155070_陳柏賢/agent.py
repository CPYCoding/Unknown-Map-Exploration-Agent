"""Unknown-Map Exploration Agent: DFS-like forward exploration + BFS to the nearest frontier."""

from collections import deque

MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


class Agent:
    def __init__(self):
        """
        Initialize the Agent.
        
        You can declare any required instance variables here,
        but the state for each new map should be reinitialized in reset().
        """
        self.reset(0, 0, (0, 0))

    def reset(self, rows, cols, start_position):
        """
        Called once before each new map.
        
        Args:
            rows: 地圖列數
            cols: 地圖欄數
            start_position: 起點座標 (row, col)
        """
        # TODO: initialize your internal map and search state.
        self.rows = rows
        self.cols = cols
        self.start_position = tuple(start_position)
        self.free = {self.start_position}
        self.walls = set()
        self.visited = set()
        self.last_move = None

    def act(self, percept):
        """
        Choose and return one of: UP, DOWN, LEFT, RIGHT.

        Args:
            percept:
                {
                    "position": (row, col),
                    "neighbors": {
                        "UP": "FREE" | "WALL" | "GOAL",
                        "DOWN": "FREE" | "WALL" | "GOAL",
                        "LEFT": "FREE" | "WALL" | "GOAL",
                        "RIGHT": "FREE" | "WALL" | "GOAL",
                    },
                }

        Returns:
            One of: UP, DOWN, LEFT, RIGHT
        """
        position = tuple(percept["position"])
        self.visited.add(position)
        self.free.add(position)

        # 1. Update the map; if the goal is adjacent, step into it
        for move, status in percept["neighbors"].items():
            cell = self._neighbor(position, move)
            if status == "GOAL":
                self.last_move = move
                return move
            if status == "FREE":
                self.free.add(cell)
            else:
                self.walls.add(cell)

        # 2. BFS to the nearest frontier and take one step toward it
        move = self._step_to_nearest_frontier(position)
        if move is None:
            # If no frontier is found, fall back to a move that doesn't hit a wall
            move = self._any_free_move(percept["neighbors"])
        self.last_move = move
        return move

    # ---------- Helper methods ----------

    def _neighbor(self, cell, move):
        dr, dc = MOVES[move]
        return (cell[0] + dr, cell[1] + dc)

    def _in_bounds(self, cell):
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def _is_unknown(self, cell):
        return self._in_bounds(cell) and cell not in self.free and cell not in self.walls

    def _unknown_count(self, cell):
        """Number of unknown cells next to this cell. More than 0 means it is a frontier."""
        return sum(self._is_unknown(self._neighbor(cell, move)) for move in MOVES)

    def _move_order(self):
        """Order for expanding neighbours in BFS: the last move comes first, so the agent keeps going straight (DFS-like)."""
        if self.last_move is None:
            return list(MOVES)
        return [self.last_move] + [m for m in MOVES if m != self.last_move]

    def _step_to_nearest_frontier(self, start):
        """
        BFS from start, only through cells known to be free.
        Return the first step toward the nearest frontier; on a tie, pick the one with the most unknown neighbours.
        """
        order = self._move_order()
        first_move = {start: None}   # the first step used to reach each cell
        queue = deque([(start, 0)])
        best = None
        best_dist = None

        while queue:
            cell, dist = queue.popleft()
            if best_dist is not None and dist > best_dist:
                break  # finished the nearest ring; no need to search further

            if cell != start:
                unknown = self._unknown_count(cell)
                if unknown > 0:
                    best_dist = dist
                    if best is None or unknown > best[0]:
                        best = (unknown, first_move[cell])
                    continue

            for move in order:
                nxt = self._neighbor(cell, move)
                if nxt in self.free and nxt not in first_move:
                    first_move[nxt] = move if cell == start else first_move[cell]
                    queue.append((nxt, dist + 1))

        return None if best is None else best[1]

    def _any_free_move(self, neighbors):
        for move in self._move_order():
            if neighbors[move] != "WALL":
                return move
        return "UP"
