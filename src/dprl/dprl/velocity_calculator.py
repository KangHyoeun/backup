#!/usr/bin/env python3
"""
velocity_calculator.py: 
    Class for calculating the velocity of the USV.     

Author: Hyo-Eun Kang 
Data  : 2024.10.
"""
import rclpy
import numpy as np
import utm
import scipy
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from wamv_msgs.msg import NavigationType
from std_msgs.msg import Float64, Header
from sensor_msgs.msg import NavSatFix, Imu

from .lib.gnc_tool import ssa, DTR, RTD

class VelocityCalculator(Node):
    def __init__(self):
        super().__init__('navigation_node')

        self.prev_time = None
        self.prev_position = None
        self.psi = 0.0  # 초기 헤딩
        self.dt = 1/20

        # 초기 목표 위치 설정
        self.goal_x = 0.0
        self.goal_y = 0.0

        self.u = 0.0
        self.v = 0.0
        self.r = 0.0  # 초기 yaw rate
        self.w = 0.0  # 초기 쿼터니언의 w 값

        self.origin_e = 284480.02
        self.origin_n = 6266152.97

        # Subscriber
        self.goal_subscriber      = self.create_subscription(PoseStamped, '/vrx/stationkeeping/goal', self.goal_callback, 10)
        self.gps_fix_subscriber   = self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix',  self.gps_fix_callback,   20)
        self.imu_data_subscriber  = self.create_subscription(Imu,       '/wamv/sensors/imu/imu/data', self.imu_data_callback,  100)
        # Publisher       
        # /navigation
        self.navigation_publisher = self.create_publisher(NavigationType, '/navigation', 20)
        self.wpx_publisher        = self.create_publisher(Float64, '/x_waypoint', 10)
        self.wpy_publisher        = self.create_publisher(Float64, '/y_waypoint', 10)

        # Set timer
        self.timer = self.create_timer(self.dt, self.publish_navigation_msg)

    def get_yaw_from_quaternion(self, quat):
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        _, _, yaw = r.as_euler('xyz', degrees=False)
        return yaw

    def transform_to_body_frame(self, vx, vy, yaw):
        u = vx * np.cos(yaw) + vy * np.sin(yaw)
        v = -vx * np.sin(yaw) + vy * np.cos(yaw)
        return u, v

    def imu_data_callback(self, msg):
        # IMU에서 쿼터니언을 받아 yaw 각도로 변환
        self.psi = self.get_yaw_from_quaternion(msg.orientation)
        # IMU에서 yaw rate(z축 각속도) 가져오기
        self.r = msg.angular_velocity.z
        # IMU에서 쿼터니언의 w 값 저장
        self.w = msg.orientation.w

    def gps_fix_callback(self, msg):
        # GPS 좌표를 UTM 좌표로 변환
        e, n, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        current_position = np.array([e - self.origin_e, n - self.origin_n])

        self.x, self.y = current_position[0], current_position[1]

        # 이전 위치가 존재하는 경우 속도 계산
        if self.prev_position is not None:
            dt = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - self.prev_time
            dx = current_position[0] - self.prev_position[0]
            dy = current_position[1] - self.prev_position[1]
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                self.u, self.v = self.transform_to_body_frame(vx, vy, self.psi)

        # 현재 위치와 시간 저장
        self.prev_position = current_position
        self.prev_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def goal_callback(self, msg):
        self.goal_data = msg
        e, n, _, _ = utm.from_latlon(msg.pose.position.x, msg.pose.position.y)

        self.goal_x, self.goal_y = e - self.origin_e, n - self.origin_n

    def publish_navigation_msg(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        msg = NavigationType()
        msg.header = header
        msg.x = float(round(self.x, 4))
        msg.y = float(round(self.y, 4))
        msg.psi = float(round(ssa(self.psi * RTD), 4))
        msg.u = float(round(self.u, 4))
        msg.v = float(round(self.v, 4))
        msg.r = float(round(self.r, 4))
        msg.w = float(round(self.w, 4))

        self.navigation_publisher.publish(msg)

        self.wpx_publisher.publish(Float64(data=self.goal_x))
        self.wpy_publisher.publish(Float64(data=self.goal_y))

def main(args=None):
    rclpy.init(args=args)
    try:
        navigation_node = VelocityCalculator()
        navigation_node.get_logger().info('NAVIGATION NODE')        
        try:
            rclpy.spin(navigation_node)
        except KeyboardInterrupt:
            navigation_node.get_logger().info('Keyboard Interrupt (SIGINT)')
        finally:
            navigation_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()