"""
src/environment/grid.py
-----------------------
Grille 2D NxM représentant l'environnement de simulation.
Gère les positions valides, les sorties, les voisins et
les distances (Manhattan, BFS).
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set, Dict
from collections import deque
import numpy as np


Position = Tuple[int, int]


class Grid:
    """
    Grille 2D discrète pour la simulation Mario Escape.

    Attributs
    ---------
    rows, cols : dimensions
    exits      : ensemble des cases de sortie
    """

    def __init__(self, rows: int, cols: int, exits: List[Position]):
        if rows < 2 or cols < 2:
            raise ValueError(f"Grille trop petite: {rows}x{cols} (minimum 2x2)")
        self.rows = rows
        self.cols = cols
        self.exits: Set[Position] = set(exits)
        self._bfs_cache: Dict[Position, Dict[Position, int]] = {}

    # ------------------------------------------------------------------
    # Propriétés de base
    # ------------------------------------------------------------------

    def is_valid(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_exit(self, pos: Position) -> bool:
        return pos in self.exits

    def neighbors(self, pos: Position) -> List[Position]:
        """Retourne les 4 voisins valides (haut, bas, gauche, droite)."""
        r, c = pos
        candidates = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        return [p for p in candidates if self.is_valid(p)]

    def all_positions(self) -> List[Position]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    @staticmethod
    def manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def bfs_distance(self, src: Position, dst: Position) -> Optional[int]:
        """Distance BFS (chemin le plus court sur la grille)."""
        if src == dst:
            return 0
        visited = {src}
        queue = deque([(src, 0)])
        while queue:
            pos, dist = queue.popleft()
            for nb in self.neighbors(pos):
                if nb == dst:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return None  # pas de chemin (ne devrait pas arriver sur grille ouverte)

    def bfs_all_distances(self, src: Position) -> Dict[Position, int]:
        """Distances BFS depuis src vers toutes les cases (mis en cache)."""
        if src in self._bfs_cache:
            return self._bfs_cache[src]
        dist_map: Dict[Position, int] = {src: 0}
        queue = deque([src])
        while queue:
            pos = queue.popleft()
            for nb in self.neighbors(pos):
                if nb not in dist_map:
                    dist_map[nb] = dist_map[pos] + 1
                    queue.append(nb)
        self._bfs_cache[src] = dist_map
        return dist_map

    def shortest_path(self, src: Position, dst: Position) -> List[Position]:
        """Retourne le chemin BFS (liste de positions) de src à dst."""
        if src == dst:
            return [src]
        parent: Dict[Position, Optional[Position]] = {src: None}
        queue = deque([src])
        found = False
        while queue and not found:
            pos = queue.popleft()
            for nb in self.neighbors(pos):
                if nb not in parent:
                    parent[nb] = pos
                    if nb == dst:
                        found = True
                        break
                    queue.append(nb)
        if not found:
            return []
        # Reconstruction
        path = []
        cur: Optional[Position] = dst
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def nearest_exit(self, pos: Position) -> Tuple[Optional[Position], float]:
        """Retourne la sortie la plus proche et sa distance BFS."""
        best_pos = None
        best_dist = float("inf")
        for exit_pos in self.exits:
            d = self.bfs_distance(pos, exit_pos)
            if d is not None and d < best_dist:
                best_dist = d
                best_pos = exit_pos
        return best_pos, best_dist

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def render(
        self,
        mario_pos: Optional[Position] = None,
        monster_pos: Optional[Position] = None,
    ) -> str:
        symbols = {
            "empty": "·",
            "exit": "★",
            "mario": "M",
            "monster": "X",
            "both": "!",
        }
        rows_str = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                pos = (r, c)
                if pos == mario_pos and pos == monster_pos:
                    row.append(symbols["both"])
                elif pos == mario_pos:
                    row.append(symbols["mario"])
                elif pos == monster_pos:
                    row.append(symbols["monster"])
                elif self.is_exit(pos):
                    row.append(symbols["exit"])
                else:
                    row.append(symbols["empty"])
            rows_str.append(" ".join(row))
        border = "+" + "-" * (self.cols * 2 - 1) + "+"
        inner = "\n".join(f"|{r}|" for r in rows_str)
        return f"{border}\n{inner}\n{border}"

    def __repr__(self) -> str:
        return f"Grid({self.rows}x{self.cols}, exits={self.exits})"
