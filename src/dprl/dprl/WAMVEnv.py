#!/usr/bin/env python3

import numpy as np
import time 

from dprl.lib.gnc_tool import saturation, ssa, DTR, RTD

import gymnasium as gym 
from gymnasium import spaces

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from scipy.stats import multivariate_normal

# 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 폰트 패밀리
matplotlib.rcParams['font.size'] = 12  # 기본 폰트 크기

import math
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WAMVEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, dT: float, target_position: np.ndarray, max_steps=200):
        super(WAMVEnv, self).__init__()
        self.dT = dT
        self.target_position = target_position
        self.max_steps = max_steps
        self.current_step = 0
        self.history = {'x': [], 'y': [], 'rewards': [], 'heading': []}
        logging.info("Environment initialized.")

        # 상태 초기화
        self.state = np.zeros(4)

        # Initial subscriber data
        self.desired_psi       = None
        self.desired_u         = None
        self.error_psi         = None
        self.error_u           = None
        self.error_x           = None
        self.error_y           = None
        self.distance          = None
        self.x_waypoint        = self.target_position[0]
        self.y_waypoint        = self.target_position[1]

        # Action space: [pwm(port), pwm(stbd), azimuth angle] # Normalize action space to [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64)

        # Actual action bounds
        self.delta_psi_bounds     = (-200.0, 200.0)
        self.delta_u_bounds       = (-200.0, 200.0)
        self.thruster_pwm_bounds  = (0.0, 2300.0)
        self.azimuth_angle_bounds = (np.deg2rad(-60), np.deg2rad(60))

        # Observation space: [distance, error_psi, error_x, error_y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def step(self, action):
        logging.info(f"Step action received: {action}")
        self.current_step += 1

        # Rescale action from [-1, 1] to actual action bounds
        delta_psi     = (action[0] * (self.delta_psi_bounds[1] - self.delta_psi_bounds[0]) / 2.0 + (self.delta_psi_bounds[1] + self.delta_psi_bounds[0]) / 2.0)
        delta_u       = (action[1] * (self.delta_u_bounds[1] - self.delta_u_bounds[0]) / 2.0 + (self.delta_u_bounds[1] + self.delta_u_bounds[0]) / 2.0)
        # thruster_pwm  = (action[2] * (self.thruster_pwm_bounds[1] - self.thruster_pwm_bounds[0]) / 2.0 + (self.thruster_pwm_bounds[1] + self.thruster_pwm_bounds[0]) / 2.0)
        # azimuth_angle = (action[3] * (self.azimuth_angle_bounds[1] - self.azimuth_angle_bounds[0]) / 2.0 + (self.azimuth_angle_bounds[1] + self.azimuth_angle_bounds[0]) / 2.0)
        
        logging.info(f"delta_psi: {delta_psi}, delta_u: {delta_u}")
        # logging.info(f"pwm: {thruster_pwm}, angle: {azimuth_angle}")

        # 이미 Control 노드로부터 상태는 update_state()를 통해 갱신되었음
        # self.state는 최신 상태를 반영
        self.position = np.array([self.state[0], self.state[1]])
        self.x, self.y = self.state[0], self.state[1]
        self.psi = self.state[2]

        # Calculate distance and heading to target
        self.distance    = np.linalg.norm(self.target_position - self.position)
        self.desired_psi = np.arctan2(self.y_waypoint - self.y, self.x_waypoint - self.x)
        self.error_psi   = np.deg2rad(ssa(self.desired_psi*RTD - self.psi))
        self.error_x     = abs(self.x_waypoint - self.x)
        self.error_y     = abs(self.y_waypoint - self.y)

        # Observation
        observation = np.array([self.distance, self.error_psi, self.error_x, self.error_y])

        # Reward
        reward = self.calculate_reward()

        # Done condition
        terminated = bool(self.distance < 3.0 and abs(self.error_psi) < np.deg2rad(10))  # Consider terminated if within 1 meter and 5 degrees of target
        truncated  = bool(self.current_step >= self.max_steps)

        done = terminated or truncated  # Combine conditions for done

        # Record history for rendering
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['rewards'].append(reward)
        self.history['heading'].append(self.psi)

        logging.info(f"Step completed. Position: {self.position}, Reward: {reward}, Done: {done}")

        return observation, reward, done, truncated, {}
    
    def calculate_reward(self):
        reward = 0
        if self.distance < 5.0:
            reward += 1000.0
        if abs(self.error_psi) < np.deg2rad(5):
            reward += 500.0
        reward += -self.distance
        reward += -self.error_x
        reward += -self.error_y
        reward += -abs(self.error_psi)
        return reward
    
    def update_state(self, state_data):
        # navigation_data를 사용하여 환경 상태 업데이트
        self.state = state_data
        self.x, self.y = self.state[0], self.state[1]
        self.psi, self.u = self.state[2], self.state[3]

        # 목표 지점에 대한 방위각 계산
        self.desired_psi = np.arctan2(self.y_waypoint - self.y, self.x_waypoint - self.x)


    def get_observation(self):
        # 에이전트에게 제공할 관측값 리턴
        if self.desired_psi is None:
            self.desired_psi = np.arctan2(self.y_waypoint - self.y, self.x_waypoint - self.x)  # 기본값 계산

        if self.distance is None:
            self.distance = 0.0  # 기본값 설정
            
        self.error_psi = np.deg2rad(ssa(self.desired_psi * RTD - self.psi))
        self.error_x   = abs(self.x_waypoint - self.x)
        self.error_y   = abs(self.y_waypoint - self.y)
        observation = np.array([self.distance, self.error_psi, self.error_x, self.error_y])
            
        # 관측값 로그 추가
        logging.info(f"Observation: {observation}")
        return observation
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 초기 상태 설정 (필요에 따라 수정 가능)
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.u = 0.0

        # 상태를 정의하고 초기화
        self.state = np.array([self.x, self.y, self.psi, self.u])
        self.history = {'x': [], 'y': [], 'rewards': [], 'heading': []}

        self.current_step = 0
        self.distance    = np.linalg.norm(self.target_position - np.array([self.x, self.y]))
        self.desired_psi = np.arctan2(self.y_waypoint - self.y, self.x_waypoint - self.x)
        self.error_psi   = np.deg2rad(ssa(self.desired_psi * RTD - self.psi))
        self.error_x     = abs(self.x_waypoint - self.x)
        self.error_y     = abs(self.y_waypoint - self.y)
        
        observation = np.array([self.distance, self.error_psi, self.error_x, self.error_y])

        logging.info("Environment reset.")
        
        return observation, {} 


    def render(self, render_mode='human', save_path='./plots', env_index=None):
        if render_mode == 'human':
            logging.info("Rendering the environment.")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

    
    def close(self):
        pass