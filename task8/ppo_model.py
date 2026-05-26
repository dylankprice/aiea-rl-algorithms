
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


#PPO actor critic using equation 1 for actor and equation 9 for critic

class ActorCritic(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(2592, 256), nn.ReLU(),
        )
        self.actor = nn.Linear(256, nb_actions)
        self.critic = nn.Linear(256, 1)


    def forward(self, x):
        h = self.head(x)
        return self.actor(h), self.critic(h)

class Environments:
    def __init__(self, nb_actors):
        self.nb_actors = nb_actors
        self.envs = [self._make_env() for _ in range(nb_actors)]
        self.observations = [None] * nb_actors
        self.done = [False] * nb_actors
        self.total_rewards = [0.0] * nb_actors
        for i in range(nb_actors):
            self.reset_env(i)

    def __len__(self):
        return self.nb_actors

    def reset_env(self, env_id):
        self.total_rewards[env_id] = 0.0
        obs, _ = self.envs[env_id].reset()
        self.observations[env_id] = obs
        self.done[env_id] = False

    def step(self, env_id, action):
        obs, reward, terminated, truncated, info = self.envs[env_id].step(int(action))
        done = terminated or truncated
        self.total_rewards[env_id] += reward
        self.observations[env_id] = obs
        self.done[env_id] = done
        return obs, reward, done, info

    def _make_env(self):
        env = gym.make("CarRacing-v3", render_mode="rgb_array",
                       lap_complete_percent=0.95, domain_randomize=False, continuous=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)   # This wrapper will keep track of cumulative rewards and episode lengths
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)        
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env


def obs_to_tensor(obs, device):
    return torch.from_numpy(np.array(obs) / 255.).float().unsqueeze(0).to(device)


def PPO(envs, actorcritic, T=128, K=3, batch_size=256, gamma=0.99,
        gae_lambda=0.95, vf_coeff=1.0, ent_coeff=0.01, nb_iterations=2000, device='cuda'):

    optimizer = torch.optim.Adam(actorcritic.parameters(), lr=2.5e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1., end_factor=0., total_iters=nb_iterations)

    writer = SummaryWriter()
    global_step = 0
    N = len(envs)
    max_reward = -np.inf
    episode_rewards = []


    smoothed = []

    for iteration in tqdm(range(nb_iterations)):

        
        buf_obs    = torch.zeros((N, T, 4, 84, 84), device=device)
        buf_acts   = torch.zeros((N, T), dtype=torch.long, device=device)
        buf_lps    = torch.zeros((N, T), device=device)
        buf_vals   = torch.zeros((N, T+1), device=device)
        buf_rews   = torch.zeros((N, T), device=device)
        buf_dones  = torch.zeros((N, T), device=device)

        with torch.no_grad():
            for env_id in range(N):
                for t in range(T):
                    obs = obs_to_tensor(envs.observations[env_id], device)
                    logits, value = actorcritic(obs)
                    dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                    action = dist.sample()

                    _, reward, done, _ = envs.step(env_id, action.item())

                    buf_obs[env_id, t]   = obs.squeeze(0)
                    buf_acts[env_id, t]  = action
                    buf_lps[env_id, t]   = dist.log_prob(action)
                    buf_vals[env_id, t]  = value.squeeze()
                    buf_rews[env_id, t]  = np.sign(reward)  # reward clipping
                    buf_dones[env_id, t] = float(done)

                    if done:
                        episode_rewards.append(envs.total_rewards[env_id])
                        if envs.total_rewards[env_id] > max_reward:
                            max_reward = envs.total_rewards[env_id]
                            torch.save(actorcritic.state_dict(), f"actorcritic_{max_reward:.0f}.pt")
                        envs.reset_env(env_id)

                # bootstrap value for last state
                last_obs = obs_to_tensor(envs.observations[env_id], device)
                _, last_val = actorcritic(last_obs)
                buf_vals[env_id, T] = last_val.squeeze()

        
        advantages = torch.zeros((N, T), device=device)
        with torch.no_grad():
            for env_id in range(N):
                gae = 0.
                for t in reversed(range(T)):
                    not_done = 1. - buf_dones[env_id, t]

                    #Equation 11 and 12 for 
                    delta = buf_rews[env_id, t] + gamma * buf_vals[env_id, t+1] * not_done - buf_vals[env_id, t]
                    gae = delta + gamma * gae_lambda * not_done * gae


                    advantages[env_id, t] = gae

        
        flat_adv  = advantages.reshape(-1)
        flat_obs  = buf_obs.reshape(-1, 4, 84, 84)
        flat_acts = buf_acts.reshape(-1)
        flat_lps  = buf_lps.reshape(-1)
        flat_vals = buf_vals[:, :T].reshape(-1)

        loader = DataLoader(TensorDataset(flat_adv, flat_obs, flat_acts, flat_lps, flat_vals),
                            batch_size=batch_size, shuffle=True)

        clip_eps = 0.1 

        for _ in range(K):
            for b_adv, b_obs, b_act, b_old_lp, b_old_val in loader:
                logits, value = actorcritic(b_obs)
                value = value.squeeze(-1)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(b_act)

                ratio = torch.exp(log_prob - b_old_lp)
                returns = b_adv + b_old_val

                # policy loss from equation 7
                p_loss = -torch.min(
                    ratio * b_adv,
                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                ).mean()

                # value loss (clipped)
                v_clipped = b_old_val + torch.clamp(value - b_old_val, -clip_eps, clip_eps)
                v_loss = torch.max(
                    F.mse_loss(returns, value, reduction='none'),
                    F.mse_loss(returns, v_clipped, reduction='none')
                ).mean()

                # total loss from equation 9
                loss = p_loss + vf_coeff * v_loss - ent_coeff * dist.entropy().mean()

                optimizer.zero_grad() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(actorcritic.parameters(), 0.5) 
                optimizer.step() # update weights
                # remove p_losses, v_losses, entropies lists entirely
                writer.add_scalar("Loss/policy", p_loss.item(), global_step)
                writer.add_scalar("Loss/value", v_loss.item(), global_step)
                writer.add_scalar("Loss/entropy", dist.entropy().mean().item(), global_step)
                writer.add_scalar("Reward/mean", np.mean(episode_rewards), iteration)
                global_step += 1

        scheduler.step() 


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    envs = Environments(nb_actors=2)
    actorcritic = ActorCritic(envs.envs[0].action_space.n).to(device)
    PPO(envs, actorcritic, device=device)
