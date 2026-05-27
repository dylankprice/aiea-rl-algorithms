import gymnasium as gym
from model import DQNetwork
import torch
from train import obs_to_tensor, select_action

env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
env = gym.wrappers.RecordEpisodeStatistics(env)  
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env)        
env = gym.wrappers.FrameStackObservation(env, 4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


weights = torch.load("racecar_model.pt")
network = DQNetwork(env.action_space.n).to(device)
network.load_state_dict(weights)


obs, _ = env.reset()
state = obs_to_tensor(obs, device)

nb_actions = env.action_space.n
nb_steps = 10000
for t in range(nb_steps):
    action = select_action(state, network, 0, nb_actions, device)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    next_state = obs_to_tensor(obs, device)
    state = next_state

    if done:
        break

            



