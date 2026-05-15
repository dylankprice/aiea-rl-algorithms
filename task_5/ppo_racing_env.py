import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

model = PPO("CnnPolicy", vec_env, verbose=1,  tensorboard_log="./ppo_car_racing_tb/", device="cuda:0")
model.learn(total_timesteps=25000)
model.save("/home/ubuntu/persistent/racing/ppo_racing")

del model # remove to demonstrate saving and loading

model = PPO.load("/home/ubuntu/persistent/racing/ppo_racing")

obs = vec_env.reset()[0]
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = vec_env.step(action)
    vec_env.render()
