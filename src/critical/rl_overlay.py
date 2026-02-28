import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# ============================================================
# REDE NEURAL
# ============================================================

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# DQN AGENT
# ============================================================

class RLOverlayAgent:

    def __init__(
        self,
        state_dim,
        actions=[0.3, 0.5, 0.7, 1.0],
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        memory_size=10000,
        batch_size=64,
        device="cpu"
    ):

        self.actions = actions
        self.action_dim = len(actions)
        self.state_dim = state_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.device = torch.device(device)

        self.q_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, self.action_dim).to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state):

        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state)
        return torch.argmax(q_values).item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(
            np.array([b[0] for b in batch])
        ).to(self.device)

        actions = torch.LongTensor(
            np.array([b[1] for b in batch])
        ).to(self.device)

        rewards = torch.FloatTensor(
            np.array([b[2] for b in batch])
        ).to(self.device)

        next_states = torch.FloatTensor(
            np.array([b[3] for b in batch])
        ).to(self.device)

        dones = torch.FloatTensor(
            np.array([b[4] for b in batch])
        ).to(self.device)

        q_values = self.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        next_q_values = self.target_net(next_states)
        next_q_value = torch.max(next_q_values, dim=1)[0]

        target = rewards + self.gamma * next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_exposure(self, state):
        action_index = self.select_action(state)
        return self.actions[action_index], action_index