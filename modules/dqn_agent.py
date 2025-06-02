import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .dqn_model import DQN
from .replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.memory = ReplayBuffer(10000)
        
        self.q_network = DQN(input_shape, n_actions).to(self.device)
        self.target_network = DQN(input_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss_fn(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def pretrain_on_demos(self, demo_states, demo_actions):
        if not demo_states or not demo_actions:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(demo_states)).to(self.device)
        actions = torch.LongTensor(np.array(demo_actions)).to(self.device)
        
        # Pre-train for a few epochs
        for _ in range(10):  # Number of pre-training epochs
            q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            targets = q_values.detach() + 0.1  # Supervised learning with small margin
            loss = self.loss_fn(q_values, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.update_target()