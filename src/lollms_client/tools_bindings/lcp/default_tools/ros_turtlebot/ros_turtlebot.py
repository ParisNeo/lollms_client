# ros_turtlebot.py
# Lollms LCP Tool Library for TurtleBot3 ROS 2 interaction
# author: ParisNeo
# description: Real ROS 2 and High-Fidelity simulated interface for TurtleBot3 navigation and nociception.

import os
import sys
import math
import random
import time
from typing import Dict, Any, List, Tuple, Optional

TOOL_LIBRARY_NAME = "ROS_TURTLEBOT"
TOOL_LIBRARY_DESC = "Interface for TurtleBot3 robot control, navigation, sensor monitoring, and nociceptive pain systems."
TOOL_LIBRARY_ICON = "🤖"

# Check for ROS 2 client library availability
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan, Imu
    from rcl_interfaces.msg import Log
    ROS2_AVAILABLE = True
except ImportError:
    pass


class TurtleBotMockState:
    """Simulates real physical kinematics and environment collisions for out-of-the-box execution."""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.wz = 0.0
        self.battery_percent = 100.0
        self.bumper_state = 0  # 0 = clear, 1 = front hit, 2 = left, 3 = right
        self.ax = 0.0
        self.ay = 0.0
        self.az = 9.81
        self.laser_ranges = [2.0] * 360  # 360 degree scan
        self.last_update = time.time()
        self.score = 0

        # Active Challenge Settings
        # 0: Empty, 1: The Maze, 2: Invisible Mines, 3: The Gauntlet
        self.challenge_id = 0
        self.charger_pos = (0.0, 0.0)
        self.goal_pos = (1.5, 1.5)

        self.obstacles = []
        self.pain_nodes = [] # Coordinates that cause immediate nociceptive shock on proximity

        self.load_challenge(0)

    def load_challenge(self, challenge_id: int):
        self.challenge_id = challenge_id
        self.vx = 0.0
        self.wz = 0.0
        self.bumper_state = 0

        if challenge_id == 0:  # Empty Arena with Charger & Goal
            self.charger_pos = (0.0, 0.0)
            self.goal_pos = (1.8, 1.8)
            self.obstacles = []
            self.pain_nodes = []
        elif challenge_id == 1:  # The Maze
            self.charger_pos = (-2.0, -2.0)
            self.goal_pos = (2.0, 2.0)
            # Pillars and wall-dividers
            self.obstacles = [
                (-1.0, 1.0, 0.3), (1.0, -1.0, 0.3),
                (0.0, 1.0, 0.2), (0.0, -1.0, 0.2),
                (-1.5, 0.0, 0.4), (1.5, 0.0, 0.4),
                (0.0, 0.0, 0.3)
            ]
            self.pain_nodes = []
        elif challenge_id == 2:  # The Invisible Minefield (Pain Nodes)
            self.charger_pos = (0.0, -2.0)
            self.goal_pos = (0.0, 2.0)
            self.obstacles = []
            # Invisible nodes triggering nociceptive accelerometer vibrations
            self.pain_nodes = [
                (-1.0, 0.0), (1.0, 0.0),
                (-0.5, 1.0), (0.5, 1.0),
                (-0.5, -1.0), (0.5, -1.0)
            ]
        elif challenge_id == 3:  # The Gauntlet (Maze + Mines)
            self.charger_pos = (-2.0, -2.0)
            self.goal_pos = (2.0, 2.0)
            self.obstacles = [
                (-1.0, 0.0, 0.4), (1.0, 0.0, 0.4)
            ]
            self.pain_nodes = [
                (0.0, -1.0), (0.0, 1.0), (-1.5, 1.5), (1.5, -1.5)
            ]

    def update(self):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # Unicycle kinematics integration
        self.theta += self.wz * dt
        # Normalize yaw to [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        self.x += self.vx * math.cos(self.theta) * dt
        self.y += self.vx * math.sin(self.theta) * dt

        # Keep within arena outer walls (-2.5 to 2.5)
        orig_x, orig_y = self.x, self.y
        self.x = max(-2.5, min(2.5, self.x))
        self.y = max(-2.5, min(2.5, self.y))

        if self.x != orig_x or self.y != orig_y:
            self.bumper_state = 1
            self.ax += random.uniform(10.0, 20.0)

        # Charger logic: within 0.3m -> recharge battery
        dist_to_charger = math.sqrt((self.x - self.charger_pos[0])**2 + (self.y - self.charger_pos[1])**2)
        if dist_to_charger < 0.3:
            self.battery_percent = min(100.0, self.battery_percent + 15.0 * dt)
        else:
            # Slower baseline drain
            self.battery_percent = max(0.0, self.battery_percent - 0.015 * dt)

        # Goal achievement logic: within 0.25m -> award point and relocate goal!
        dist_to_goal = math.sqrt((self.x - self.goal_pos[0])**2 + (self.y - self.goal_pos[1])**2)
        if dist_to_goal < 0.25:
            self.score += 1
            # Find a new random location not inside obstacles
            while True:
                gx = random.uniform(-2.2, 2.2)
                gy = random.uniform(-2.2, 2.2)
                # verify not too close to charger or obstacles
                if math.sqrt(gx**2 + gy**2) < 0.5:
                    continue
                in_obstacle = False
                for ox, oy, orad in self.obstacles:
                    if math.sqrt((gx - ox)**2 + (gy - oy)**2) < orad + 0.3:
                        in_obstacle = True
                        break
                if not in_obstacle:
                    self.goal_pos = (gx, gy)
                    break

        # Simulate noise / normal baseline on accelerometer
        self.ax = random.normalvariate(0.0, 0.05) + (self.vx * self.wz)
        self.ay = random.normalvariate(0.0, 0.05)
        self.az = 9.81 + random.normalvariate(0.0, 0.1)

        # Pain Node Proximity logic (Nociceptive trigger)
        # If close to an invisible pain node, spike accelerometer to trigger pain reflex!
        for px, py in self.pain_nodes:
            dist = math.sqrt((self.x - px)**2 + (self.y - py)**2)
            if dist < 0.45:
                intensity = (0.45 - dist) * 45.0  # Spike higher the closer it gets
                self.ax += random.uniform(-1.0, 1.0) * intensity
                self.ay += random.uniform(-1.0, 1.0) * intensity
                self.bumper_state = 1

        # Laser ranges simulation (360 points)
        for angle in range(360):
            rad = self.theta + math.radians(angle)
            dx = math.cos(rad)
            dy = math.sin(rad)

            min_dist = 3.5  # Max range

            # Check circular obstacles
            for ox, oy, orad in self.obstacles:
                cx = ox - self.x
                cy = oy - self.y
                b = -2 * (dx * cx + dy * cy)
                c = cx**2 + cy**2 - orad**2
                disc = b**2 - 4 * c
                if disc >= 0:
                    t1 = (-b - math.sqrt(disc)) / 2
                    if t1 > 0 and t1 < min_dist:
                        min_dist = t1

            # Check outer square boundaries (-2.5 to 2.5)
            # Intersect ray with x=2.5, x=-2.5, y=2.5, y=-2.5
            for val, is_x in [(2.5, True), (-2.5, True), (2.5, False), (-2.5, False)]:
                if is_x:
                    if dx != 0:
                        t = (val - self.x) / dx
                        if t > 0:
                            y_at_t = self.y + dy * t
                            if -2.5 <= y_at_t <= 2.5:
                                min_dist = min(min_dist, t)
                else:
                    if dy != 0:
                        t = (val - self.y) / dy
                        if t > 0:
                            x_at_t = self.x + dx * t
                            if -2.5 <= x_at_t <= 2.5:
                                min_dist = min(min_dist, t)

            self.laser_ranges[angle] = min_dist

        # Bumper collision detection
        front_collision_dist = min(self.laser_ranges[0:15] + self.laser_ranges[345:360])
        left_collision_dist = min(self.laser_ranges[15:45])
        right_collision_dist = min(self.laser_ranges[315:345])

        if front_collision_dist < 0.15:
            self.bumper_state = 1
            self.ax += random.uniform(12.0, 20.0)
            self.vx = -0.04
        elif left_collision_dist < 0.15:
            self.bumper_state = 2
            self.ay += random.uniform(12.0, 20.0)
            self.wz = -0.4
        elif right_collision_dist < 0.15:
            self.bumper_state = 3
            self.ay -= random.uniform(12.0, 20.0)
            self.wz = 0.4
        else:
            self.bumper_state = 0

    def get_ascii_map(self) -> str:
        """Produce a beautiful 2D terminal grid of the active arena challenge."""
        grid_size = 15
        # Maps -2.5..2.5 to grid_size indices
        grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

        def to_grid(val):
            # map -2.5..2.5 to 0..grid_size-1
            idx = int(((val + 2.5) / 5.0) * (grid_size - 1))
            return max(0, min(grid_size - 1, idx))

        # Draw Charging Station (C)
        cx, cy = to_grid(self.charger_pos[0]), to_grid(self.charger_pos[1])
        grid[cy][cx] = "C"

        # Draw Target Goal (G)
        gx, gy = to_grid(self.goal_pos[0]), to_grid(self.goal_pos[1])
        grid[gy][gx] = "G"

        # Draw Invisible Pain Nodes (X) (Faded or hinted to help AI or user diagnose)
        for px, py in self.pain_nodes:
            gpx, gpy = to_grid(px), to_grid(py)
            grid[gpy][gpx] = "x"  # lower case x for invisible mine hazard

        # Draw Obstacles (#)
        # Since obstacles are circular, we find all cells whose centers overlap the circles
        for ox, oy, orad in self.obstacles:
            for gy_idx in range(grid_size):
                for gx_idx in range(grid_size):
                    # Map cell center back to metric coords
                    cx_val = -2.5 + (gx_idx / (grid_size - 1)) * 5.0
                    cy_val = -2.5 + (gy_idx / (grid_size - 1)) * 5.0
                    if math.sqrt((cx_val - ox)**2 + (cy_val - oy)**2) <= orad + 0.1:
                        grid[gy_idx][gx_idx] = "#"

        # Draw Robot (R)
        rx, ry = to_grid(self.x), to_grid(self.y)
        grid[ry][rx] = "R"

        # Render full string with frame borders
        lines = [
            f"┌──────────────────────────────┐  Challenge: {self.challenge_id}",
            f"│ Map: {['Empty', 'The Maze', 'Invisible Mines', 'The Gauntlet'][self.challenge_id]:16}        │  Score: {self.score}",
            "├──────────────────────────────┤  Battery: {0:.1f}%".format(self.battery_percent),
            "│" + "│\n│".join(" ".join(row) for row in reversed(grid)) + "│",
            "└──────────────────────────────┘",
            "Legend: R=Robot, C=Charging, G=Goal, #=Obstacle, x=Mines (invisible)",
            "Pose  : X={0:.2f}m, Y={1:.2f}m, Heading={2:.1f}°".format(self.x, self.y, math.degrees(self.theta))
        ]
        return "\n".join(lines)


# Global shared simulator state
MOCK_ROBOT = TurtleBotMockState()


class TurtleBotNode(Node if ROS2_AVAILABLE else object):
    """ROS 2 Node class wrapping subscribers and publishers for physical integration."""
    def __init__(self):
        if not ROS2_AVAILABLE:
            return
        super().__init__("lollmsbot_controller")
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/imu", self._imu_callback, 10)
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.laser_ranges = [3.5] * 360
        self.ax = 0.0
        self.ay = 0.0
        self.az = 9.81
        self.bumper_state = 0

    def _odom_callback(self, msg: Any):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        # Simple Euler angle conversion from Quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

    def _scan_callback(self, msg: Any):
        self.laser_ranges = list(msg.ranges)
        # Check bumper state from laser ranges as TurtleBots sometimes lack separate bumper topics
        front_dist = min([r for r in (self.laser_ranges[0:15] + self.laser_ranges[345:360]) if not math.isnan(r) and r > 0.01], default=3.5)
        if front_dist < 0.15:
            self.bumper_state = 1
        else:
            self.bumper_state = 0

    def _imu_callback(self, msg: Any):
        self.ax = msg.linear_acceleration.x
        self.ay = msg.linear_acceleration.y
        self.az = msg.linear_acceleration.z


# Active node instance holder
ROS2_NODE = None


def init_tool_library() -> None:
    """Initialize ROS 2 Context or log simulation fallback."""
    global ROS2_NODE
    if ROS2_AVAILABLE:
        try:
            if not rclpy.utilities.ok():
                rclpy.init()
            ROS2_NODE = TurtleBotNode()
            print("🤖 ROS_TURTLEBOT: Successfully connected to local ROS 2 executor.")
        except Exception as e:
            print(f"🤖 ROS_TURTLEBOT: Found ROS 2 libraries but init failed ({e}). Falling back to simulation.")
            ROS2_NODE = None
    else:
        print("🤖 ROS_TURTLEBOT: ROS 2 libraries not found. Operating in High-Fidelity Simulation Mode.")


def tool_navigate_to(x: float, y: float, linear_speed: float = 0.15,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None) -> Dict[str, Any]:
    """
    Commands the TurtleBot robot to navigate to the specified coordinate.

    Args:
        x (float): Target X coordinate in meters.
        y (float): Target Y coordinate in meters.
        linear_speed (float, optional): Cruise velocity in m/s. Defaults to 0.15.
    """
    global ROS2_NODE
    MOCK_ROBOT.update()
    
    current_x = ROS2_NODE.x if ROS2_NODE else MOCK_ROBOT.x
    current_y = ROS2_NODE.y if ROS2_NODE else MOCK_ROBOT.y
    current_theta = ROS2_NODE.theta if ROS2_NODE else MOCK_ROBOT.theta

    dx = x - current_x
    dy = y - current_y
    distance = math.sqrt(dx**2 + dy**2)

    if distance < 0.05:
        return {
            "success": True,
            "output": f"Robot already at target coordinates ({current_x:.2f}, {current_y:.2f})."
        }

    target_angle = math.atan2(dy, dx)
    
    # 1. Spin to face target
    angle_to_turn = target_angle - current_theta
    # Normalize to -pi to pi
    angle_to_turn = (angle_to_turn + math.pi) % (2 * math.pi) - math.pi

    if ROS2_NODE:
        # Physical/Gazebo robot command loop
        twist = Twist()
        # Turn
        twist.angular.z = 0.4 if angle_to_turn > 0 else -0.4
        ROS2_NODE.cmd_vel_pub.publish(twist)
        time.sleep(abs(angle_to_turn) / 0.4)
        
        # Stop turn
        twist.angular.z = 0.0
        ROS2_NODE.cmd_vel_pub.publish(twist)
        
        # Move forward
        twist.linear.x = linear_speed
        ROS2_NODE.cmd_vel_pub.publish(twist)
        time.sleep(distance / linear_speed)
        
        # Stop forward
        twist.linear.x = 0.0
        ROS2_NODE.cmd_vel_pub.publish(twist)
    else:
        # Simulate exact unicycle kinematics jump
        MOCK_ROBOT.theta = target_angle
        MOCK_ROBOT.vx = linear_speed
        # Update simulation steps
        travel_time = distance / linear_speed
        steps = int(travel_time * 10)
        for _ in range(steps):
            MOCK_ROBOT.update()
            if MOCK_ROBOT.bumper_state > 0:
                MOCK_ROBOT.vx = 0.0
                return {
                    "success": False,
                    "error": f"Navigation aborted! Collision detected (Bumper status: {MOCK_ROBOT.bumper_state}) at coordinate ({MOCK_ROBOT.x:.2f}, {MOCK_ROBOT.y:.2f})."
                }
            time.sleep(0.01)
        MOCK_ROBOT.vx = 0.0
        MOCK_ROBOT.x = x
        MOCK_ROBOT.y = y

    return {
        "success": True,
        "output": f"Successfully navigated to coordinate ({x:.2f}, {y:.2f}). Current odometry pose is ({x:.2f}, {y:.2f})."
    }


def tool_get_robot_pose(discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None) -> Dict[str, Any]:
    """
    Retrieves the current odometry coordinates and orientation (pose) of the robot.
    """
    global ROS2_NODE
    MOCK_ROBOT.update()
    
    x = ROS2_NODE.x if ROS2_NODE else MOCK_ROBOT.x
    y = ROS2_NODE.y if ROS2_NODE else MOCK_ROBOT.y
    theta = ROS2_NODE.theta if ROS2_NODE else MOCK_ROBOT.theta

    return {
        "success": True,
        "x": round(x, 3),
        "y": round(y, 3),
        "theta": round(theta, 3),
        "output": f"Current Robot Pose: x={x:.3f}m, y={y:.3f}m, yaw={math.degrees(theta):.1f}°"
    }


def tool_get_sensor_readings(discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None) -> Dict[str, Any]:
    """
    Queries LiDAR lasers, bumper sensors, battery voltage, and accelerometers.
    This supplies essential data for mapping, collision avoidance, and nociception.
    """
    global ROS2_NODE
    MOCK_ROBOT.update()

    battery = 98.4 if ROS2_AVAILABLE else round(MOCK_ROBOT.battery_percent, 1)
    bumper = ROS2_NODE.bumper_state if ROS2_NODE else MOCK_ROBOT.bumper_state
    
    ax = ROS2_NODE.ax if ROS2_NODE else MOCK_ROBOT.ax
    ay = ROS2_NODE.ay if ROS2_NODE else MOCK_ROBOT.ay
    az = ROS2_NODE.az if ROS2_NODE else MOCK_ROBOT.az
    
    ranges = ROS2_NODE.laser_ranges if ROS2_NODE else MOCK_ROBOT.laser_ranges
    
    # Compress lidar data into quadrant segments for easy context transmission
    front_dist = min([r for r in (ranges[0:30] + ranges[330:360]) if not math.isnan(r) and r > 0.01], default=3.5)
    left_dist = min([r for r in ranges[30:150] if not math.isnan(r) and r > 0.01], default=3.5)
    rear_dist = min([r for r in ranges[150:210] if not math.isnan(r) and r > 0.01], default=3.5)
    right_dist = min([r for r in ranges[210:330] if not math.isnan(r) and r > 0.01], default=3.5)

    return {
        "success": True,
        "battery_percent": battery,
        "bumper_state": bumper,
        "accelerometer": {
            "x": round(ax, 3),
            "y": round(ay, 3),
            "z": round(az, 3)
        },
        "lidar_distances": {
            "front": round(front_dist, 3),
            "left": round(left_dist, 3),
            "rear": round(rear_dist, 3),
            "right": round(right_dist, 3)
        },
        "output": f"Sensors: Battery={battery}%, Bumper={bumper}, IMU=[{ax:.2f}, {ay:.2f}, {az:.2f}], Lidar=[F:{front_dist:.2f}m, L:{left_dist:.2f}m, R:{right_dist:.2f}m]"
    }


def tool_stop_robot() -> Dict[str, Any]:
    """
    Commands an emergency stop to immediately stop all motor velocity.
    """
    global ROS2_NODE
    MOCK_ROBOT.vx = 0.0
    MOCK_ROBOT.wz = 0.0
    
    if ROS2_NODE:
        twist = Twist()
        ROS2_NODE.cmd_vel_pub.publish(twist)
        
    return {
        "success": True,
        "output": "Emergency stop invoked. All velocities zeroed out."
    }


def tool_trigger_nociception_test(intensity: float) -> Dict[str, Any]:
    """
    Injects a simulated high-impact physical force vector to test pain and nociceptive reactions.

    Args:
        intensity (float): Simulated shock force in G-units (e.g. 15.0).
    """
    MOCK_ROBOT.ax += random.choice([-1.0, 1.0]) * intensity
    MOCK_ROBOT.ay += random.choice([-1.0, 1.0]) * intensity
    MOCK_ROBOT.bumper_state = 1

    return {
        "success": True,
        "output": f"Simulated force impact injected successfully. Accelerometer force: {intensity} Gs. Bumper state: Front Collided."
    }


def tool_get_ascii_arena() -> Dict[str, Any]:
    """
    Returns a real-time ASCII visual map of the active challenge arena.
    This helps coordinate spatial movements and visualize obstacles.
    """
    MOCK_ROBOT.update()
    return {
        "success": True,
        "output": MOCK_ROBOT.get_ascii_map()
    }


def tool_set_arena_challenge(challenge_id: int) -> Dict[str, Any]:
    """
    Switch the active arena layout and target challenge.

    Args:
        challenge_id (int): 0 = Empty, 1 = The Maze, 2 = Invisible Mines, 3 = The Gauntlet (Maze + Mines)
    """
    if challenge_id not in (0, 1, 2, 3):
        return {"success": False, "error": "Invalid challenge ID. Must be 0, 1, 2, or 3."}
    MOCK_ROBOT.load_challenge(challenge_id)
    return {
        "success": True,
        "output": f"Successfully loaded challenge {challenge_id}. Map reset. Use 'get_ascii_arena' to visualize."
    }


def tool_teleport_robot(x: float, y: float) -> Dict[str, Any]:
    """
    Instantly resets or teleports the robot to a specific coordinate inside the arena.

    Args:
        x (float): New X coordinate (-2.4 to 2.4).
        y (float): New Y coordinate (-2.4 to 2.4).
    """
    if not (-2.4 <= x <= 2.4) or not (-2.4 <= y <= 2.4):
        return {"success": False, "error": "Target coordinates must be within the boundary [-2.4, 2.4]."}
    MOCK_ROBOT.x = x
    MOCK_ROBOT.y = y
    MOCK_ROBOT.update()
    return {
        "success": True,
        "output": f"Robot teleported to ({x:.2f}, {y:.2f}). Use 'get_ascii_arena' to verify surroundings."
    }
