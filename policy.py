import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax

"""
class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, s):
        x = self.dense1(s)
        x = relu(x)
        x = self.dense2(x)
        return softmax(x, dim=1)
"""

class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, s):
        x = self.dense1(s)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)
        return softmax(x, dim=1)
    
class BackwardPolicy:
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.size = int(state_dim**0.5)
        self.device = "cpu"
    
    def __call__(self, s):
        idx = s.argmax(-1)
        at_top_edge = idx < self.size
        at_left_edge = (idx > 0) & (idx % self.size == 0)
        
        probs = torch.ones(len(s), self.num_actions).to(self.device)
        probs[at_left_edge, 1] = 0
        probs[at_top_edge, 0] = 0
        probs[:, -1] = 0 # disregard termination
        
        #normalize
        probs = probs / probs.sum(-1, keepdim=True)
        
        return probs
    
    def to(self, *args, **kwargs):
        self.device = args[0]
        return self