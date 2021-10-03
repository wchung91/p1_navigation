import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #Check if GPU is available

class Agent():
    #Intialize Agent
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #Initalize Q-QNetworks and optimizer
        self.qnetwork_local = QNetwork(state_size,action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        #Intialize memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        #time step
        self.t_step = 0

    #Each step adds data to experience/Reply Buffer
    #Agent learns every UPDATE_EVERY step
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1 )% UPDATE_EVERY
        if self.t_step == 0 :
            if len(self.memory) > BATCH_SIZE:
                #print("Updated the network")
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    #Agent pick an action
    #Action could be random action or best action
    def act(self, state, eps=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #gets action values from network
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # based on eps random or best action is taken
        if random.random() > eps:
            #print("Using the best action")
            return np.argmax(action_values.cpu().data.numpy())
        else:
            #print("Testing out new move")
            return random.choice(np.arange(self.action_size))

    #Update/Learn network
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) #compute the q_target

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #soft update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    #soft update the target network based on the local network
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)

#ReplayBuffer to contain experiences
class ReplayBuffer:
    #Intialize ReplayBuffer
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    # Adds data to the replay buffer
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    # Picks sample from the buffer
    def sample(self):
        #experiences = random.sample(self.memory, k=self.batch_size)
        #states = torch.from_numpy(np.vstack([e.state for experiences in e if e is not None])).to(device)
        #actions = torch.from_numpy(np.vstack([e.action for experiences in e if e is not None])).to(device)
        #rewards = torch.from_numpy(np.vstack([e.reward for experiences in e if e is not None])).to(device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for experiences in e if e is not None])).to(device)
        #dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).to(device)
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
