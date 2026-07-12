import numpy as np

from src.environment.dynamic_obstacles import (
    LinearObstacle,
    WaypointObstacle,
    RandomWalkObstacle,
)

from src.environment.obstacle_manager import ObstacleManager


def main():
    print("=" * 50)
    print("Testing LinearObstacle")
    print("=" * 50)

    linear = LinearObstacle(
        obstacle_id="L1",
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        radius=1.0,
    )

    print("Before:", linear.position)

    linear.update(1.0)

    print("After :", linear.position)

    print()

    print("=" * 50)
    print("Testing WaypointObstacle")
    print("=" * 50)

    waypoint = WaypointObstacle(
        obstacle_id="W1",
        position=np.array([0.0, 0.0]),
        radius=1.0,
        speed=2.0,
        waypoints=[
            np.array([10.0, 0.0]),
            np.array([10.0, 10.0]),
        ],
    )

    print("Before:", waypoint.position)

    waypoint.update(1.0)

    print("After :", waypoint.position)

    print()

    print("=" * 50)
    print("Testing RandomWalkObstacle")
    print("=" * 50)

    random = RandomWalkObstacle(
        obstacle_id="R1",
        position=np.array([50.0, 50.0]),
        radius=1.0,
        speed=1.5,
        seed=42,
    )

    print("Before:", random.position)

    random.update(1.0)

    print("After :", random.position)

    print()

    print("=" * 50)
    print("Testing ObstacleManager")
    print("=" * 50)

    manager = ObstacleManager()

    manager.add_obstacle(linear)
    manager.add_obstacle(waypoint)
    manager.add_obstacle(random)

    print(manager)

    print("Obstacle count:", manager.obstacle_count())
    print("Active count :", manager.active_obstacle_count())

    manager.update(0.5)

    print()

    result = manager.nearest_obstacle(np.array([0.0, 0.0]))

    print("Nearest obstacle:")
    print(result)

    print()

    prediction = manager.predict_all_positions(2.0)

    print("Future positions:")

    for p in prediction:
        print(p)

    print()

    collision = manager.check_collision(np.array([1.0, 0.0]))

    print("Collision Query:")
    print(collision)

    print()

    print("All tests completed successfully!")


if __name__ == "__main__":
    main()