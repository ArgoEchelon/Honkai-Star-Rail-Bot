import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array([e[0] for e in experiences]))
        actions = torch.LongTensor(np.array([e[1] for e in experiences]))
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences]))
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences]))
        dones = torch.FloatTensor(np.array([e[4] for e in experiences]).astype(np.uint8))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network for approximating Q-values."""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class DQNAgent:
    """DQN Agent that learns to play the game."""
    
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99          # Discount factor
        self.learning_rate = 0.001
        self.epsilon = 1.0         # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.buffer_size = 100000
        self.batch_size = 64
        self.update_target_freq = 10  # Update target network every N episodes
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is in evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = ReplayBuffer(self.buffer_size)
        
        # Training metrics
        self.training_step = 0
    
    def select_action(self, state, evaluation=False):
        """Select an action using epsilon-greedy policy."""
        if evaluation or random.random() > self.epsilon:
            # Exploit: select best action according to the policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
        else:
            # Explore: select a random action
            return random.randrange(self.action_size)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """Update policy network based on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        curr_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values with the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and perform gradient descent
        loss = self.loss_fn(curr_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network by copying weights from the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filename):
        """Save the agent's models."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load_model(self, filename):
        """Load the agent's models."""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']