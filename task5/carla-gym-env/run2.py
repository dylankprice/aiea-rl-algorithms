import gymnasium as gym
import gym_carla
import carla
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def main():
    params = {
        'number_of_vehicles': 1,
        'number_of_walkers': 0,
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': False,
        'discrete_acc': [-3.0, 0.0, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln*',
        'port': 2000,
        'town': 'Town03',
        'max_time_episode': 1000,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'display_route': False,
    }

    env = gym.make('carla-v1', params=params)
    writer = SummaryWriter(log_dir="./tensorboard_EVAL/")

    for episode in range(20):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            action = env.action_space.sample()  # random actions, no model needed
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        print(f"Episode {episode} | Reward: {episode_reward:.2f}")
        writer.add_scalar("Eval/episode_reward", episode_reward, episode)
        writer.add_scalar("Eval/episode_length", episode_steps, episode)

    writer.close()
    env.close()

if __name__ == '__main__':
    main()