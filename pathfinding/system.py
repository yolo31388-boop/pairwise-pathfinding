"""路径寻路导航 - 含6个bug"""
from dataclasses import dataclass, field
import heapq

@dataclass
class GridMap:
    width: int
    height: int
    obstacles: set = field(default_factory=set)

class PathfindingSystem:
    def __init__(self, grid_map: GridMap):
        self.grid = grid_map
        self.offline_paths: dict = {}  # bug6: 不随地图更新

    def find_path(self, start: tuple, end: tuple) -> list:
        # bug1: BFS不支持动态障碍物，且大地图慢
        queue = [start]
        visited = {start}
        parent = {}
        while queue:
            current = queue.pop(0)
            if current == end:
                path = []
                while current in parent:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                return path[::-1]
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = current[0]+dx, current[1]+dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if (nx, ny) not in visited and (nx, ny) not in self.grid.obstacles:
                        visited.add((nx, ny))
                        parent[(nx, ny)] = current
                        queue.append((nx, ny))
        return []

    def smooth_path(self, path: list) -> list:
        # bug2: 直线连接不检查碰撞
        if len(path) <= 2:
            return path
        return [path[0], path[-1]]

    def build_navmesh(self, agent_radius: float) -> set:
        # bug3: 不考虑NPC半径
        return set(self.grid.obstacles)

    def hierarchical_pathfind(self, start: tuple, end: tuple) -> list:
        # bug4: 直接用细粒度
        return self.find_path(start, end)

    def calculate_avoidance(self, position: tuple, others: list) -> tuple:
        # bug5: 未实现
        return position

    def find_path_offline(self, start: tuple, end: tuple) -> list:
        # bug6: 预计算不失效
        key = (start, end)
        if key in self.offline_paths:
            return self.offline_paths[key]
        path = self.find_path(start, end)
        self.offline_paths[key] = path
        return path
