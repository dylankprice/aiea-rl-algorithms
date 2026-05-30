import gymnasium as gym
import gym_carla
import carla
from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter #used for tensoboard logging

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

    env = gym.make('carla-v0', params=params)
    model = SAC.load("SAC_dist")

    # Added a writer for eval logging
    writer = SummaryWriter(log_dir="./tensorboard_EVAL/")

    obs, info = env.reset()
    episode = 0
    episode_reward = 0
    episode_steps = 0

    while episode < 20:  # run 20 eval episodes
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1

        if terminated or truncated:
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {episode_steps}")
            writer.add_scalar("Eval/episode_reward", episode_reward, episode)
            writer.add_scalar("Eval/episode_length", episode_steps, episode)
            episode_reward = 0
            episode_steps = 0
            episode += 1
            obs, info = env.reset()

    writer.close()
    env.close()

if __name__ == '__main__':
    main()