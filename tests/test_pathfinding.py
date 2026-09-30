"""路径寻路导航 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathfinding.system import PathfindingSystem, GridMap

class TestAStar:
    def test_finds_path_around_obstacle(self):
        grid = GridMap(5, 5, obstacles={(2, 0), (2, 1), (2, 2)})
        ps = PathfindingSystem(grid)
        path = ps.find_path((0, 0), (4, 0))
        assert len(path) > 0
        assert path[-1] == (4, 0)

class TestSmoothCollision:
    def test_smoothed_path_does_not_cross_obstacle(self):
        grid = GridMap(5, 5, obstacles={(2, 2)})
        ps = PathfindingSystem(grid)
        path = [(0, 0), (1, 1), (3, 3), (4, 4)]
        smoothed = ps.smooth_path(path)
        assert (2, 2) not in smoothed

class TestNavmeshRadius:
    def test_narrow_passage_blocked_for_large_agent(self):
        grid = GridMap(5, 5, obstacles={(1, 0), (1, 2)})
        ps = PathfindingSystem(grid)
        navmesh = ps.build_navmesh(agent_radius=2.0)
        assert (1, 1) in navmesh

class TestDynamicObstacle:
    def test_replans_when_obstacle_appears(self):
        grid = GridMap(5, 5)
        ps = PathfindingSystem(grid)
        path1 = ps.find_path((0, 0), (4, 0))
        grid.obstacles.add((2, 0))
        path2 = ps.find_path((0, 0), (4, 0))
        assert path1 != path2

class TestAvoidance:
    def test_avoids_other_agents(self):
        grid = GridMap(5, 5)
        ps = PathfindingSystem(grid)
        result = ps.calculate_avoidance((2, 2), [(2, 2)])
        assert result != (2, 2)

class TestOfflineInvalidation:
    def test_offline_path_invalidated_on_map_change(self):
        grid = GridMap(5, 5)
        ps = PathfindingSystem(grid)
        ps.find_path_offline((0, 0), (4, 0))
        grid.obstacles.add((2, 0))
        path = ps.find_path_offline((0, 0), (4, 0))
        assert (2, 0) not in path
