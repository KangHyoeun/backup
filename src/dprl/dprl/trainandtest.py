from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from dprl.WAMVEnv import WAMVEnv

# 환경 생성
env = WAMVEnv(dT=0.1, target_position=np.array([50, 50]))

# 모델 설정 및 학습
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_wamv_tensorboard/")

# 평가 콜백 설정
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=10000,
                             deterministic=True, render=False)

# 학습 실행
model.learn(total_timesteps=500000, callback=eval_callback)

# 학습된 모델 저장
model.save('ppo_wamv_trained_model')

# 모델을 사용한 예측
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()