import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .dqn_model import DuelingDistributionalDQN
from .replay_buffer import PrioritizedReplayBuffer

class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.n_atoms = 51
        self.v_min = -10
        self.v_max = 10
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.memory = PrioritizedReplayBuffer(10000)
        self.n_steps = 3
        self.transitions = []

        self.q_network = DuelingDistributionalDQN(input_shape, n_actions, self.n_atoms, self.v_min, self.v_max).to(self.device)
        self.target_network = DuelingDistributionalDQN(input_shape, n_actions, self.n_atoms, self.v_min, self.v_max).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network.get_q_values(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        if len(self.transitions) >= self.n_steps or done:
            self._store_multi_step_transition()

    def _store_multi_step_transition(self):
        if not self.transitions:
            return
        state = self.transitions[0][0]
        action = self.transitions[0][1]
        n_step_reward = 0
        for i in range(len(self.transitions)):
            n_step_reward += (self.gamma ** i) * self.transitions[i][2]
        next_state = self.transitions[-1][3]
        done = self.transitions[-1][4]
        self.memory.add(state, action, n_step_reward, next_state, done)
        if done:
            self.transitions = []
        elif len(self.transitions) > self.n_steps:
            self.transitions.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        curr_dist = self.q_network(states)
        curr_dist = curr_dist[range(self.batch_size), actions]

        with torch.no_grad():
            next_q_values = self.q_network.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=-1)
            next_dist = self.target_network(next_states)
            next_dist = next_dist[range(self.batch_size), next_actions]

            t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.n_steps) * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.n_atoms).to(self.device)

            proj_dist = torch.zeros_like(curr_dist)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        loss = -torch.sum(proj_dist * torch.log(curr_dist + 1e-6), dim=-1) * is_weights

        priorities = loss.detach().abs().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)

        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())