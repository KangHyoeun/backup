#!/usr/bin/env python3
"""
rl_control.py: 
    Class for the DRL DP control of the USV.     

Author: Hyo-Eun Kang 
Data  : 2024.10.
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO  # PPO 알고리즘을 예로 사용
from dprl.WAMVEnv import WAMVEnv
import rclpy
import numpy as np
import time 
import scipy
from scipy.spatial.transform import Rotation as R

from rclpy.time import Time
from rclpy.node import Node

from wamv_msgs.msg import NavigationType, ControlType, GuidanceType
from std_msgs.msg import Float64

from .lib.gnc_tool import saturation, ssa, DTR, RTD


class Control(Node):
    def __init__(self, wamv_env):
        super().__init__('control_node')

        self.agent = PPO.load("/home/hyo/dp_ws/src/dprl/dprl/ppo_wamv_initial_model.zip")
        
        # Initial subscriber data
        self.navigation_data   = None
        self.target_position   = np.array([50.0, 50.0])
        self.target_psi        = 0.0 # deg
        self.desired_psi       = None
        self.desired_u         = None
        self.error_psi         = None
        self.error_u           = None
        self.distance          = None
        self.state             = None
        self.position          = None
        self.x_waypoint        = self.target_position[0]
        self.y_waypoint        = self.target_position[1]
        
        # Subscriber
        self.navigation_subscriber = self.create_subscription(NavigationType, '/navigation', self.navigation_callback, 20)

        # Publisher 
        self.guidance_publisher     = self.create_publisher(GuidanceType, '/guidance', 10)
        self.control_publisher      = self.create_publisher(ControlType, '/control', 10)
        self.left_pos_publisher     = self.create_publisher(Float64, '/wamv/thrusters/left/pos',     10)
        self.left_thrust_publisher  = self.create_publisher(Float64, '/wamv/thrusters/left/thrust',  10)
        self.right_pos_publisher    = self.create_publisher(Float64, '/wamv/thrusters/right/pos',    10)
        self.right_thrust_publisher = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        self.wamv_env = wamv_env

        # Allocation class input data
        self.dt       = 0.1

        # Set timer
        self.timer = self.create_timer(self.dt, self.publish_control_msg)

    def navigation_callback(self, msg):
        self.navigation_data = msg
        self.x   = self.navigation_data.x
        self.y   = self.navigation_data.y
        self.psi = self.navigation_data.psi
        self.u   = self.navigation_data.u

        self.position = np.array([self.x, self.y])
        self.state    = np.array([self.x, self.y, self.psi, self.u])

        # WAMV 환경에 새로운 상태 업데이트
        self.wamv_env.update_state(self.state)

        # 로그 추가 - 상태 확인
        self.get_logger().info(f'Updated state: {self.state}')

    def step(self):
        return self.state
        
    def publish_control_msg(self):
        # self.get_logger().info(self.navigation_data)

        if self.navigation_data is None or self.state is None:
            self.get_logger().info(f'check the navigation data and state update!', once = True)
            return

        # main
        self.get_logger().info('WAM-V RL CONTROL START', once = True)

        # 환경으로부터 관측 데이터를 받아와서 PPO 에이전트가 사용할 수 있게 합니다.
        observation = self.wamv_env.get_observation()

        # RL agent로부터 action을 받아옴
        action, _ = self.agent.predict(observation, deterministic=True)

        # 환경의 스텝 수행
        obs, reward, done, truncated, _ = self.wamv_env.step(action)

        delta_psi, delta_u = action
        self.get_logger().info(f"Delta Psi: {delta_psi}, Delta U: {delta_u}")

        # 원하는 각도와 속도 계산
        x,y = self.position
        self.desired_u = 0.5

        self.distance    = np.linalg.norm(self.target_position - self.position)
        self.desired_psi = np.arctan2(self.y_waypoint - y, self.x_waypoint - x)
        self.error_psi   = np.deg2rad(ssa(self.desired_psi*RTD - self.psi))
        self.error_u     = self.desired_u - self.u
        self.error_x     = abs(self.x_waypoint - self.x)
        self.error_y     = abs(self.y_waypoint - self.y)

        self.thruster_pwm = 1500.0

        # 제한 설정 및 PWM 계산
        thruster_pwm_port = self.thruster_pwm - delta_u
        thruster_pwm_stbd = self.thruster_pwm - delta_u

        # azimuth angle 초기값 설정 (중립 타각 테스트)
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
        msg2.desired_psi = round(self.desired_psi, 4)
        msg2.desired_u   = round(self.desired_u, 2)
        msg2.error_psi   = round(self.error_psi, 4)
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

        # 환경이 종료 상태이면 환경 리셋
        if done:
            self.wamv_env.reset()

def main(args=None):
    rclpy.init(args=args)
    wamv_env = WAMVEnv(dT=0.1, target_position=np.array([50.0, 50.0]))
    try:
        control_node = Control(wamv_env)
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
