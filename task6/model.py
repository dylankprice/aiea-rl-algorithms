import torch
import torch.nn as nn


# Shared CNN for actor and critic
class ActorCritic(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(4096, 256), nn.ReLU(),  # 4096 for 96x96 input
        )

        # Continuous actor: outputs mean for each action dimension
        self.actor_mean    = nn.Linear(256, nb_actions)
        # log_std as a learned parameter (not input-dependent, simpler and stable)
        self.actor_log_std = nn.Parameter(torch.zeros(nb_actions))

        # Critic: scalar value estimate
        self.critic = nn.Linear(256, 1)

        # Small init so policy starts near uniform
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, x):
        h    = self.head(x)
        mean = self.actor_mean(h)
        std  = self.actor_log_std.exp().expand_as(mean)
        return mean, std, self.critic(h)

    def get_action(self, x):
        mean, std, value = self.forward(x)
        dist   = torch.distributions.Normal(mean, std)
        raw    = dist.rsample()                          # reparameterized sample
        action = torch.tanh(raw)                         # squash to (-1, 1)

        # rescale gas and brake from (-1, 1) to (0, 1)
        action_env = action.clone()
        action_env[..., 1] = (action[..., 1] + 1) / 2  # gas
        action_env[..., 2] = (action[..., 2] + 1) / 2  # brake

        # log prob with tanh correction
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)

        # action_env -> sent to environment
        # action     -> stored in buffer for evaluate()
        return action_env, action, log_prob, value.squeeze(-1)

    def evaluate(self, x, action):
        mean, std, value = self.forward(x)
        dist     = torch.distributions.Normal(mean, std)
        raw      = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))  # inverse tanh
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy