

import random
from collections import deque
import torch.nn as nn
import torch.nn.functional as F

#1. DQNetwork class          ← the CNN
# ReplayBuffer class       ← stores experience tuples
# select_action function   ← epsilon-greedy logic
 # train function           ← the main loop (Algorithm 1)
class DQNetwork(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(2592, 256), nn.ReLU(),
        )
        self.fc = nn.Linear(256, nb_actions)

    def forward(self, x):
        return self.fc(self.conv(x))
    

        

