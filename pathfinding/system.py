"""路径寻路导航"""
from dataclasses import dataclass, field
import heapq
import itertools
import math

@dataclass
class GridMap:
    width: int
    height: int
    obstacles: set = field(default_factory=set)

class PathfindingSystem:
    def __init__(self, grid_map: GridMap):
        self.grid = grid_map
        self.offline_paths: dict = {}  # key -> (map_signature, path)

    def find_path(self, start: tuple, end: tuple) -> list:
        # A*，每次调用读取当前障碍物，动态障碍出现时重新调用即重规划
        return self._astar(
            start, end, (self.grid.width, self.grid.height),
            lambda p: p not in self.grid.obstacles,
        )

    def _astar(self, start: tuple, end: tuple, dims: tuple, walkable) -> list:
        width, height = dims
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            return []
        if not (0 <= end[0] < width and 0 <= end[1] < height):
            return []
        if start != end and not walkable(end):
            return []
        counter = itertools.count()
        open_heap = [(self._heuristic(start, end), next(counter), start)]
        came_from = {start: None}
        g_score = {start: 0}
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                if not walkable(neighbor):
                    continue
                tentative = g_score[current] + 1
                if tentative < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    priority = tentative + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (priority, next(counter), neighbor))
        return []

    @staticmethod
    def _heuristic(a: tuple, b: tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def smooth_path(self, path: list) -> list:
        # 视线平滑：仅当两点间直线无碰撞时才跳过中间点
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

    def _line_of_sight(self, a: tuple, b: tuple) -> bool:
        return all(cell not in self.grid.obstacles for cell in self._line_cells(a, b))

    @staticmethod
    def _line_cells(a: tuple, b: tuple) -> list:
        # 超覆盖Bresenham：对角步同时检查两侧相邻格，防止贴角穿墙
        x0, y0 = a
        x1, y1 = b
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        nx = abs(x1 - x0)
        ny = abs(y1 - y0)
        x, y = x0, y0
        cells = [(x, y)]
        err = nx - ny
        while x != x1 or y != y1:
            e2 = 2 * err
            if -ny < e2 < nx:
                cells.append((x + sx, y))
                cells.append((x, y + sy))
                err += nx - ny
                x += sx
                y += sy
            elif e2 <= -ny:
                err += nx
                y += sy
            else:
                err -= ny
                x += sx
            cells.append((x, y))
        return cells

    def build_navmesh(self, agent_radius: float) -> set:
        # 返回该半径下不可行走的格子：障碍物按NPC半径膨胀，狭窄通道会被封死
        blocked = set()
        reach = max(0, int(math.ceil(agent_radius)))
        for ox, oy in self.grid.obstacles:
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    if math.hypot(dx, dy) < agent_radius:
                        cell = (ox + dx, oy + dy)
                        if 0 <= cell[0] < self.grid.width and 0 <= cell[1] < self.grid.height:
                            blocked.add(cell)
        return blocked

    def hierarchical_pathfind(self, start: tuple, end: tuple, block_size: int = 10) -> list:
        # 层级寻路：先在粗粒度块图上搜抽象路径，再在路径走廊内细化
        width, height = self.grid.width, self.grid.height
        if width * height <= (2 * block_size) ** 2:
            return self.find_path(start, end)

        coarse_w = (width + block_size - 1) // block_size
        coarse_h = (height + block_size - 1) // block_size

        def block_cells(cx, cy):
            for x in range(cx * block_size, min((cx + 1) * block_size, width)):
                for y in range(cy * block_size, min((cy + 1) * block_size, height)):
                    yield (x, y)

        def coarse_walkable(c):
            return any(cell not in self.grid.obstacles for cell in block_cells(*c))

        coarse_start = (start[0] // block_size, start[1] // block_size)
        coarse_end = (end[0] // block_size, end[1] // block_size)
        coarse_path = self._astar(coarse_start, coarse_end, (coarse_w, coarse_h), coarse_walkable)
        if not coarse_path:
            return []

        # 抽象路径及其相邻块构成细化搜索的走廊
        allowed = {start, end}
        for cx, cy in coarse_path:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if 0 <= cx + dx < coarse_w and 0 <= cy + dy < coarse_h:
                        allowed.update(block_cells(cx + dx, cy + dy))

        path = self._astar(
            start, end, (width, height),
            lambda p: p in allowed and p not in self.grid.obstacles,
        )
        if not path:
            return self.find_path(start, end)
        return path

    def calculate_avoidance(self, position: tuple, others: list, radius: float = 2.0) -> tuple:
        # 简单避让转向：对范围内的其他单位（含移动障碍如玩家）施加斥力
        px, py = position
        steer_x = 0.0
        steer_y = 0.0
        for ox, oy in others:
            rx, ry = px - ox, py - oy
            dist = math.hypot(rx, ry)
            if dist >= radius:
                continue
            if dist == 0.0:
                # 完全重叠时沿确定方向推开
                rx, ry, dist = 1.0, 0.0, 1.0
            strength = (radius - dist) / radius
            steer_x += rx / dist * strength
            steer_y += ry / dist * strength
        if steer_x == 0.0 and steer_y == 0.0:
            return position
        return (px + steer_x, py + steer_y)

    def _map_signature(self) -> tuple:
        return (self.grid.width, self.grid.height, frozenset(self.grid.obstacles))

    def find_path_offline(self, start: tuple, end: tuple) -> list:
        # 预计算缓存按地图签名失效：地图改动后自动重算
        key = (start, end)
        signature = self._map_signature()
        cached = self.offline_paths.get(key)
        if cached is not None and cached[0] == signature:
            return list(cached[1])
        path = self.find_path(start, end)
        self.offline_paths[key] = (signature, path)
        return list(path)
