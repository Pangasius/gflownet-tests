import torch
from torch.nn.functional import one_hot
from gflownet.env import Env

class Grid(Env):
    def __init__(self, size):
        self.size = size
        self.state_dim = size**2
        self.num_actions = 3 # down, right, terminate
        self.device = "cpu"
        
    def update(self, s, actions):
        idx = s.argmax(1)
        down, right = actions == 0, actions == 1
        idx[down] = idx[down] + self.size
        idx[right] = idx[right] + 1
        return one_hot(idx, self.state_dim).float()
    
    def mask(self, s):
        mask = torch.ones(len(s), self.num_actions).to(self.device)
        idx = s.argmax(1) + 1
        right_edge = (idx > 0) & (idx % (self.size) == 0)
        bottom_edge = idx > self.size*(self.size-1)
        mask[right_edge, 1] = 0
        mask[bottom_edge, 0] = 0
        return mask
        
    def reward(self, s):
        grid = s.view(len(s), self.size, self.size)
        coord = (grid == 1).nonzero()[:, 1:].view(len(s), 2)
        R0, R1, R2 = 1e-2, 0.5, 2
        norm = torch.abs(coord / (self.size-1) - 0.5)
        R1_term = torch.prod(0.25 < norm, dim=1)
        R2_term = torch.prod((0.3 < norm) & (norm < 0.4), dim=1)
        return (R0 + R1*R1_term + R2*R2_term)
    
    def terminal_action(self, actions):
        return actions == self.num_actions - 1
    
    def terminal_state(self, s):
        return torch.zeros(len(s)).to(self.device).bool()
    
    def to_grid(self) :
        #for every possible state, compute the reward
        
        #create a grid of all possible states
        grid = torch.eye(self.state_dim).view(self.state_dim, self.size, self.size)
         
        #compute the reward for each state using the reward function
        rewards = self.reward(grid)
        
        return rewards
    
    def to(self, *args, **kwargs):
        self.device = args[0]
        return self
         