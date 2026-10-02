"""路径寻路导航"""
from dataclasses import dataclass, field
import heapq
import math


@dataclass
class GridMap:
    width: int
    height: int
    obstacles: set = field(default_factory=set)


class PathfindingSystem:
    CLUSTER_SIZE = 10

    def __init__(self, grid_map: GridMap):
        self.grid = grid_map
        self.offline_paths: dict = {}
        self._offline_version = None

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid.width and 0 <= y < self.grid.height

    def _walkable(self, x: int, y: int) -> bool:
        return self._in_bounds(x, y) and (x, y) not in self.grid.obstacles

    def _neighbors(self, node: tuple):
        x, y = node
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if self._walkable(nx, ny):
                yield (nx, ny)

    @staticmethod
    def _heuristic(a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: tuple, end: tuple) -> list:
        # A* 搜索；每次调用都读取当前障碍物，动态障碍出现后重新调用即可重规划
        if start == end:
            return [start] if self._walkable(*start) else []
        if not self._walkable(*start) or not self._walkable(*end):
            return []
        open_heap = [(self._heuristic(start, end), 0, start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()
        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            if current in closed:
                continue
            closed.add(current)
            for neighbor in self._neighbors(current):
                tentative = g + 1
                if tentative < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    heapq.heappush(
                        open_heap,
                        (tentative + self._heuristic(neighbor, end), tentative, neighbor),
                    )
        return []

    def _line_of_sight(self, a: tuple, b: tuple) -> bool:
        # Bresenham 逐格检查直线路径上的碰撞
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if not self._walkable(x, y):
                return False
            if (x, y) == (x1, y1):
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def smooth_path(self, path: list) -> list:
        # 拉直路径，但只跳过视线内无碰撞的路段
        if len(path) <= 2:
            return list(path)
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_of_sight(path[i], path[j]):
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def build_navmesh(self, agent_radius: float) -> set:
        # 返回不可行走格集合：障碍物按 NPC 半径膨胀，狭窄通道会被正确封死
        blocked = set()
        reach = int(math.ceil(agent_radius))
        for ox, oy in self.grid.obstacles:
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    if dx * dx + dy * dy > agent_radius * agent_radius:
                        continue
                    x, y = ox + dx, oy + dy
                    if self._in_bounds(x, y):
                        blocked.add((x, y))
        return blocked

    def hierarchical_pathfind(self, start: tuple, end: tuple) -> list:
        # 大地图先在聚类网格上求粗路径，再在细粒度上逐段细化
        size = self.CLUSTER_SIZE
        if self.grid.width <= size and self.grid.height <= size:
            return self.find_path(start, end)

        def cluster_of(p):
            return (p[0] // size, p[1] // size)

        def cluster_bounds(c):
            cx, cy = c
            xs = range(cx * size, min((cx + 1) * size, self.grid.width))
            ys = range(cy * size, min((cy + 1) * size, self.grid.height))
            return xs, ys

        def cluster_walkable(c):
            xs, ys = cluster_bounds(c)
            return any((x, y) not in self.grid.obstacles for x in xs for y in ys)

        def cluster_rep(c):
            xs, ys = cluster_bounds(c)
            mx = min(c[0] * size + size // 2, self.grid.width - 1)
            my = min(c[1] * size + size // 2, self.grid.height - 1)
            best, best_d = None, math.inf
            for x in xs:
                for y in ys:
                    if (x, y) in self.grid.obstacles:
                        continue
                    d = (x - mx) ** 2 + (y - my) ** 2
                    if d < best_d:
                        best, best_d = (x, y), d
            return best

        start_c, end_c = cluster_of(start), cluster_of(end)
        if start_c == end_c:
            return self.find_path(start, end)

        # 粗粒度 A*
        open_heap = [(self._heuristic(start_c, end_c), 0, start_c)]
        came_from = {}
        g_score = {start_c: 0}
        closed = set()
        coarse_path = []
        max_cx = (self.grid.width - 1) // size
        max_cy = (self.grid.height - 1) // size
        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current == end_c:
                coarse_path = [current]
                while current in came_from:
                    current = came_from[current]
                    coarse_path.append(current)
                coarse_path.reverse()
                break
            if current in closed:
                continue
            closed.add(current)
            cx, cy = current
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nc = (cx + dx, cy + dy)
                if not (0 <= nc[0] <= max_cx and 0 <= nc[1] <= max_cy):
                    continue
                if not cluster_walkable(nc):
                    continue
                tentative = g + 1
                if tentative < g_score.get(nc, math.inf):
                    g_score[nc] = tentative
                    came_from[nc] = current
                    heapq.heappush(
                        open_heap,
                        (tentative + self._heuristic(nc, end_c), tentative, nc),
                    )
        if not coarse_path:
            return self.find_path(start, end)

        # 细粒度逐段细化，任一段失败则回退到整体 A*
        waypoints = [start]
        for c in coarse_path[1:-1]:
            rep = cluster_rep(c)
            if rep is None:
                return self.find_path(start, end)
            waypoints.append(rep)
        waypoints.append(end)

        full_path = []
        for a, b in zip(waypoints, waypoints[1:]):
            leg = self.find_path(a, b)
            if not leg:
                return self.find_path(start, end)
            full_path.extend(leg if not full_path else leg[1:])
        return full_path

    def calculate_avoidance(self, position: tuple, others: list) -> tuple:
        # 简单避让：附近有其他 agent 时，移向离它们最远的可行走邻格
        px, py = position
        close = [o for o in others
                 if (o[0] - px) ** 2 + (o[1] - py) ** 2 <= 4]
        if not close:
            return position
        candidates = [(px, py)] + list(self._neighbors(position))
        best, best_score = position, -1.0
        for cell in candidates:
            score = min((cell[0] - o[0]) ** 2 + (cell[1] - o[1]) ** 2
                        for o in close)
            if score > best_score:
                best, best_score = cell, score
        return best

    def _map_version(self) -> int:
        return hash(frozenset(self.grid.obstacles))

    def find_path_offline(self, start: tuple, end: tuple) -> list:
        # 预计算路径缓存；地图（障碍物）变化后缓存整体失效
        version = self._map_version()
        if version != self._offline_version:
            self.offline_paths.clear()
            self._offline_version = version
        key = (start, end)
        if key not in self.offline_paths:
            self.offline_paths[key] = self.find_path(start, end)
        return self.offline_paths[key]
