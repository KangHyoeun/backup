#!/usr/bin/env python3
"""
dp_control.py: 
    Class for the DP control of the USV. 원본 코드.

Author: Hyo-Eun Kang 
Data  : 2024.10.
"""
import rclpy
import numpy as np
import time 
import utm
import scipy
from scipy.spatial.transform import Rotation as R

from rclpy.time import Time
from rclpy.node import Node

from wamv_msgs.msg import NavigationType, ControlType, GuidanceType
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped

from .lib.gnc_tool import saturation, ssa, DTR, RTD
from .lib.control_tool import PIDControl, Smoother, distance_to_waypoint


class Control(Node):
    def __init__(self):
        super().__init__('control_node')
        
        # Initial subscriber data
        self.navigation_data   = None
        self.target_position   = np.zeros(2)
        self.desired_psi       = None
        self.desired_u         = None
        self.error_psi         = None
        self.error_u           = None
        self.distance          = None
        self.x_waypoint        = None
        self.y_waypoint        = None
        
        # Subscriber
        self.navigation_subscriber = self.create_subscription(NavigationType, '/navigation', self.navigation_callback, 20)
        self.wpx_subscriber        = self.create_subscription(Float64, '/x_waypoint', self.wpx_callback, 10)
        self.wpy_subscriber        = self.create_subscription(Float64, '/y_waypoint', self.wpy_callback, 10)

        # Publisher 
        # /guidance
        # /control     
        # /wamv/thrusters/left/pos
        # /wamv/thrusters/left/thrust
        # /wamv/thrusters/right/pos
        # /wamv/thrusters/right/thrust
        self.guidance_publisher     = self.create_publisher(GuidanceType, '/guidance', 10)
        self.control_publisher      = self.create_publisher(ControlType, '/control', 10)
        self.left_pos_publisher     = self.create_publisher(Float64, '/wamv/thrusters/left/pos',     10)
        self.left_thrust_publisher  = self.create_publisher(Float64, '/wamv/thrusters/left/thrust',  10)
        self.right_pos_publisher    = self.create_publisher(Float64, '/wamv/thrusters/right/pos',    10)
        self.right_thrust_publisher = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        # Allocation class input data
        self.dt         = 0.1
        
        # Get Parameter
        self.declare_parameter('kp_psi', 20.0)
        self.declare_parameter('ki_psi', 0.0)
        self.declare_parameter('kd_psi', 5.0)
        self.declare_parameter('saturation_psi', 500)
        self.declare_parameter('kp_u', 1.0)
        self.declare_parameter('ki_u', 0.0)
        self.declare_parameter('kd_u', 0.0)
        self.declare_parameter('saturation_u', 100)
        self.declare_parameter('desired_u', 0.7)
        self.declare_parameter('thruster_pwm', 1300)
        self.declare_parameter('thruster_pwm_min', 0)
        self.declare_parameter('thruster_pwm_max', 2300)
        self.declare_parameter('thruster_wn', 5.0)
        self.declare_parameter('thruster_zeta', 1.0)
        self.get_param()
        
        # Set class
        self.azimuthControl     = PIDControl()
        self.speedControl       = PIDControl()
        self.pwmSmootherPort    = Smoother(self.thruster_pwm)
        self.pwmSmootherStbd    = Smoother(self.thruster_pwm)

        # Set timer
        self.timer = self.create_timer(self.dt, self.publish_control_msg)

    def wpx_callback(self, msg):
        self.x_waypoint = msg.data
    
    def wpy_callback(self, msg):
        self.y_waypoint = msg.data

    def navigation_callback(self, msg):
        self.navigation_data = msg
        self.x   = self.navigation_data.x
        self.y   = self.navigation_data.y
        self.psi = self.navigation_data.psi
        self.u   = self.navigation_data.u
        self.r   = self.navigation_data.r

        self.position = np.array([self.x, self.y])
        
    def get_param(self):
        self.kp_psi           = self.get_parameter('kp_psi').value
        self.ki_psi           = self.get_parameter('ki_psi').value
        self.kd_psi           = self.get_parameter('kd_psi').value
        self.saturation_psi   = self.get_parameter('saturation_psi').value
        self.kp_u             = self.get_parameter('kp_u').value
        self.ki_u             = self.get_parameter('ki_u').value
        self.kd_u             = self.get_parameter('kd_u').value
        self.saturation_u     = self.get_parameter('saturation_u').value
        self.desired_u        = self.get_parameter('desired_u').value
        self.thruster_pwm     = self.get_parameter('thruster_pwm').value
        self.thruster_pwm_min = self.get_parameter('thruster_pwm_min').value
        self.thruster_pwm_max = self.get_parameter('thruster_pwm_max').value
        self.thruster_wn      = self.get_parameter('thruster_wn').value
        self.thruster_zeta    = self.get_parameter('thruster_zeta').value
        
    def publish_control_msg(self):
        # self.get_logger().info(self.navigation_data)

        if self.navigation_data is None or self.x_waypoint is None or self.y_waypoint is None:
            self.get_logger().info(f'check the goal and navigation data!', once = True)
            return
        
        self.target_position = np.array([self.x_waypoint, self.y_waypoint])

        # Get parameter
        self.get_param()
        
        # Update class
        self.azimuthControl.update(self.kp_psi, self.ki_psi, self.kd_psi)
        self.speedControl.update(self.kp_u, self.ki_u, self.kd_u)
        self.pwmSmootherPort.update(self.thruster_wn, self.thruster_zeta)
        self.pwmSmootherStbd.update(self.thruster_wn, self.thruster_zeta)

        # main
        self.get_logger().info('WAM-V CONTROL START', once = True)

        # 원하는 각도와 속도 계산
        x,y = self.position

        self.distance = np.linalg.norm(self.target_position - self.position)
        self.desired_psi = np.arctan2(self.y_waypoint - y, self.x_waypoint - x)
        self.error_psi = np.deg2rad(ssa(self.desired_psi*RTD - self.psi))
        self.error_u   = self.desired_u - self.u

        # 원하는 각도와 속도의 변화량 계산
        delta_psi = self.azimuthControl.main(self.error_psi, None, self.dt)
        delta_u = self.speedControl.main(self.error_u, None, self.dt)

        # 제한 설정 및 PWM 계산
        # thruster_pwm_port = float(saturation(self.thruster_pwm - delta_u, self.thruster_pwm_min, self.thruster_pwm_max))
        # thruster_pwm_stbd = float(saturation(self.thruster_pwm - delta_u, self.thruster_pwm_min, self.thruster_pwm_max))
        thruster_pwm_port = 1500.0
        thruster_pwm_stbd = 1500.0

        # azimuth angle 초기값 설정 (중립 타각 테스트)
        # azimuth_angle_port = float(saturation(delta_psi, np.deg2rad(-60.0), np.deg2rad(60.0)))
        # azimuth_angle_stbd = float(saturation(delta_psi, np.deg2rad(-60.0), np.deg2rad(60.0)))
        azimuth_angle_port = delta_psi
        azimuth_angle_stbd = delta_psi

        # Publish
        msg                    = ControlType()
        msg.delta_psi          = round(delta_psi, 4)
        msg.delta_u            = round(delta_u, 4)
        msg.pwm_standard       = float(round(self.thruster_pwm, 2))
        msg.thruster_pwm_port  = float(round(thruster_pwm_port, 2))
        msg.thruster_pwm_stbd  = float(round(thruster_pwm_stbd, 2))
        msg.azimuth_angle_port = float(round(np.rad2deg(azimuth_angle_port), 2))
        msg.azimuth_angle_stbd = float(round(np.rad2deg(azimuth_angle_stbd), 2))

        msg2             = GuidanceType()
        msg2.desired_psi = round(self.desired_psi*RTD, 4)
        msg2.desired_u   = round(self.desired_u, 2)
        msg2.error_psi   = round(self.error_psi*RTD, 4)
        msg2.error_u     = round(self.error_u, 2)
        msg2.distance    = round(self.distance, 2)
        msg2.x_waypoint  = self.x_waypoint
        msg2.y_waypoint  = self.y_waypoint

        self.guidance_publisher.publish(msg2)
        self.control_publisher.publish(msg)
        
        # 퍼블리시할 메시지 생성
        port_angle_msg = Float64(data = azimuth_angle_port)
        stbd_angle_msg = Float64(data = azimuth_angle_stbd) 
        port_pwm_msg = Float64(data = thruster_pwm_port)
        stbd_pwm_msg = Float64(data = thruster_pwm_stbd)

        # publish
        self.left_pos_publisher.publish(port_angle_msg)
        self.left_thrust_publisher.publish(port_pwm_msg)
        self.right_pos_publisher.publish(stbd_angle_msg)
        self.right_thrust_publisher.publish(stbd_pwm_msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        control_node = Control()
        control_node.get_logger().info('CONTROL NODE')        
        try:
            rclpy.spin(control_node)
        except KeyboardInterrupt:
            control_node.get_logger().info('Keyboard Interrupt (SIGINT)')
        finally:
            control_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
