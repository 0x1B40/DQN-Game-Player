import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.ptr = 0
        self.size = 0

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def sample(self, value):
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx - self.capacity + 1, self.tree[idx]
            left_val = self.tree[left]
            if value <= left_val:
                idx = left
            else:
                value -= left_val
                idx = left + 1

    def total(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.epsilon = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority = self.tree.sample(s)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        batch = [self.tree.data[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(indices),
            np.array(is_weights)
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx + self.tree.capacity - 1, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size