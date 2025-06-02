import torch
import torch.nn as nn
import numpy as np

class DuelingDistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_atoms=51, v_min=-10, v_max=10):
        super(DuelingDistributionalDQN, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * n_atoms)
        )
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_atoms)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size(0)
        conv_out = self.conv(x).view(batch_size, -1)
        value = self.fc_value(conv_out).view(batch_size, 1, self.n_atoms)
        advantage = self.fc_advantage(conv_out).view(batch_size, self.n_actions, self.n_atoms)
        advantage_mean = advantage.mean(1, keepdim=True)
        q_dist = value + (advantage - advantage_mean)
        q_dist = torch.softmax(q_dist, dim=-1)
        return q_dist

    def get_q_values(self, x):
        q_dist = self.forward(x)
        return (q_dist * self.support.to(x.device)).sum(dim=-1)