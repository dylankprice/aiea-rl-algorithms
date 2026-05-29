

import random
from collections import deque
import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),  #added 3rd cnn layer
            nn.Flatten(), nn.Linear(5184, 256), nn.ReLU(),
        )
        self.fc = nn.Linear(256, nb_actions)

    def forward(self, x):
        return self.fc(self.conv(x))
    

        

