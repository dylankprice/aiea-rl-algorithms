

from collections import deque
import random
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from model import DQNetwork
from torch.utils.tensorboard import SummaryWriter


device="cuda:0" if torch.cuda.is_available() else "cpu"

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

def obs_to_tensor(obs, device):
    return torch.from_numpy(np.array(obs) / 255.).float().unsqueeze(0).to(device)

# randomly picks either greedy or non greedy based on epsilon
def select_action(state, network, epsilon, nb_actions, device):
    if random.random() < epsilon:
        return random.randint(0, nb_actions - 1) 
    else:
        with torch.no_grad(): 
            q_values = network(state) 
            return torch.argmax(q_values).item() 
    

    
def train(env, network,target_network, buffer, nb_episodes, nb_steps, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, device):
    epsilon = epsilon_start
    nb_actions = env.action_space.n
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4) # create optimizer for network parameters
    writer = SummaryWriter()
    global_step = 0
    best_reward = -np.inf  # used for saving best model based on episode reward
    
    for episode in range(nb_episodes):
        obs, _ = env.reset() # reset env to get first observeration
        state = obs_to_tensor(obs, 'cpu')  
        episode_reward = 0 # track episode reward for logging

        for t in range(nb_steps):
            action = select_action(state.to(device), network, epsilon, nb_actions, device)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = obs_to_tensor(obs, 'cpu')
            buffer.push(state, action, np.clip(reward, -1, 1), next_state, done)
            state = next_state

            episode_reward += reward
            global_step += 1
            if global_step % 1000 == 0:
                target_network.load_state_dict(network.state_dict())

            if len(buffer) >= batch_size:
                minibatch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype = torch.float32).to(device)

            
                with torch.no_grad():
                    best_actions = network(next_states).argmax(dim=1, keepdim=True)
                    y_j = rewards + gamma * target_network(next_states).gather(1, best_actions).squeeze(1) * (1 - dones)  #modified to double DQN target calculation

                current_qs = network(states).gather(1, torch.LongTensor(actions).unsqueeze(1).to(device)) #get Q value for each action taken in minibatch
                loss = F.mse_loss(current_qs.squeeze(1), y_j) # compute loss between current q values and target q values
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() # update network parameters
                writer.add_scalar("Loss/train", loss.item(), global_step)

            epsilon = max(epsilon_end, epsilon * epsilon_decay) # epsilon annealing
            writer.add_scalar("Epsilon", epsilon, global_step)

        print(f"Episode {episode}/{nb_episodes} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.3f} | Best: {best_reward:.1f}")

        writer.add_scalar("Reward/episode", episode_reward, episode)
        if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(network.state_dict(), "racecar_model.pt")

    writer.close()




if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)  
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)        
    env = gym.wrappers.FrameStackObservation(env, 4)

    network = DQNetwork(env.action_space.n).to(device)

    target_network = DQNetwork(env.action_space.n).to(device)
    target_network.load_state_dict(network.state_dict())  # copy weights

    buffer = ReplayBuffer(capacity=10_000)

    

    train(env, network, target_network, buffer, 
        nb_episodes=500,
        nb_steps=1000,
        batch_size=32,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99997,
        device= device)