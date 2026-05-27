

from collections import deque
import random
# ReplayBuffer class       ← stores experience tuples
# select_action function   ← epsilon-greedy logic
 # train function           ← the main loop (Algorithm 1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
# randomly picks non greedy actions based on epsilon to ensure non determinstic behavior
def select_action(state, network, epsilon, nb_actions, device):
    if random.random() < epsilon:
        return random.randint(0, nb_actions - 1) 
    else:
        with torch.no_grad(): 
            q_values = network(state) 
            return torch.argmax(q_values).item() 
    

    
def train(env, network, buffer, nb_episodes, nb_steps, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, device):
    epsilon = epsilon_start
    
    for episode in nb_episodes:

        