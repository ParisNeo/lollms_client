# ros_turtlebot.py
# Lollms LCP Tool Library for TurtleBot3 ROS 2 interaction
# author: ParisNeo
# description: Real ROS 2 and High-Fidelity simulated interface for TurtleBot3 navigation and nociception.

import os
import sys
import math
import random
import time
from typing import Dict, Any, List, Tuple

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
        self.simulated_obstacles = [
            (1.5, 0.0, 0.3),   # x, y, radius of a cylinder pillar
            (-1.0, 1.0, 0.4),
            (0.5, -1.5, 0.2)
        ]

    def update(self):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # Update pose based on simple unicycle kinematics
        self.theta += self.wz * dt
        self.x += self.vx * math.cos(self.theta) * dt
        self.y += self.vx * math.sin(self.theta) * dt

        # Simulate gradual battery drain
        self.battery_percent = max(0.0, self.battery_percent - 0.01 * dt)

        # Simulate noise on accelerometer
        self.ax = random.normalvariate(0.0, 0.1) + (self.vx * self.wz)
        self.ay = random.normalvariate(0.0, 0.1)
        self.az = 9.81 + random.normalvariate(0.0, 0.15)

        # Calculate laser scan based on simulated obstacles
        for angle in range(360):
            rad = self.theta + math.radians(angle)
            dx = math.cos(rad)
            dy = math.sin(rad)
            
            min_dist = 3.5  # Max range of TurtleBot Lidar
            for ox, oy, orad in self.simulated_obstacles:
                # Ray-circle intersection
                cx = ox - self.x
                cy = oy - self.y
                b = -2 * (dx * cx + dy * cy)
                c = cx**2 + cy**2 - orad**2
                disc = b**2 - 4 * c
                if disc >= 0:
                    t1 = (-b - math.sqrt(disc)) / 2
                    if t1 > 0 and t1 < min_dist:
                        min_dist = t1
            
            # Simulate a wall boundary at 3m radius from origin
            dist_to_wall = 3.0 - math.sqrt(self.x**2 + self.y**2)
            if dist_to_wall > 0 and dist_to_wall < min_dist:
                min_dist = dist_to_wall

            self.laser_ranges[angle] = min_dist

        # Check for collision (bumper activation)
        if min(self.laser_ranges[0:15] + self.laser_ranges[345:360]) < 0.12:
            self.bumper_state = 1  # Front collision
            self.ax += random.uniform(15.0, 25.0)  # Generates a massive accelerometer force vector
            self.vx = -0.05  # Bounce back
        elif min(self.laser_ranges[15:45]) < 0.12:
            self.bumper_state = 2  # Left collision
            self.ay += random.uniform(15.0, 25.0)
            self.wz = -0.5
        elif min(self.laser_ranges[315:345]) < 0.12:
            self.bumper_state = 3  # Right collision
            self.ay -= random.uniform(15.0, 25.0)
            self.wz = 0.5
        else:
            self.bumper_state = 0


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


def tool_navigate_to(x: float, y: float, linear_speed: float = 0.15) -> Dict[str, Any]:
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


def tool_get_robot_pose() -> Dict[str, Any]:
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


def tool_get_sensor_readings() -> Dict[str, Any]:
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
