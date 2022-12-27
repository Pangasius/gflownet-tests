import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax

class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions, normalization):
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_actions)
        self.normalization = normalization
    
    def forward(self, s):
        norm = s * self.normalization
        
        x = self.dense1(s)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        x = self.dense3(x)
        return softmax(x, dim=1)
    
    def to(self, *args, **kwargs):
        self.normalization = self.normalization.to(args[0])
        return super().to(*args, **kwargs)
    
class BackwardPolicy:
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.device = "cpu"
    
    def __call__(self, s):
        probs = torch.ones(len(s), self.num_actions).to(self.device)
        
        #if not jumping or ducking, then can't backprop it
        probs[:, 0] = torch.where(s[:, 1].bool(), torch.zeros_like(probs[:, 0]), torch.ones_like(probs[:, 0]))
        probs[:, 1] = torch.where(s[:, 2].bool(), torch.zeros_like(probs[:, 1]), torch.ones_like(probs[:, 1]))
         
        return probs
    
    def to(self, *args, **kwargs):
        self.device = args[0]
        return self