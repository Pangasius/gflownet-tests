import torch
from torch.nn.functional import one_hot
from gflownet.env import Env

class Grid(Env):
    def __init__(self, size):
        self.size = size
        self.state_dim = size**2
        self.num_actions = 9 # up-left, up, up-right, left, right, down-left, down, down-right, terminate
        
    def update(self, s, actions):
        idx = s.argmax(1)
        up, down, left, right = actions == 0, actions == 1, actions == 2, actions == 3
        idx[up] = idx[up] - self.size
        idx[down] = idx[down] + self.size
        idx[left] = idx[left] - 1
        idx[right] = idx[right] + 1
        return one_hot(idx, self.state_dim).float()
    
    def mask(self, s):
        mask = torch.ones(len(s), self.num_actions)
        idx = s.argmax(1) + 1
        left_edge = (idx > 0) & (idx % (self.size) == 1)
        right_edge = (idx > 0) & (idx % (self.size) == 0)
        top_edge = idx < self.size
        bottom_edge = idx > self.size*(self.size-1)
        mask[left_edge, 2] = 0
        mask[right_edge, 0] = 0
        mask[top_edge, 1] = 0
        mask[bottom_edge, 3] = 0
        return mask
    
    def reward(self, s):
        grid = s.view(len(s), self.size, self.size)
        coord = (grid == 1).nonzero()[:, 1:].view(len(s), 2)
        R0, R1, R2 = 1e-2, 0.5, 2
        norm = torch.abs(coord / (self.size-1) - 0.5)
        R1_term = torch.prod(0.25 < norm, dim=1)
        R2_term = torch.prod((0.3 < norm) & (norm < 0.4), dim=1)
        return (R0 + R1*R1_term + R2*R2_term)
    
    def to_grid(self) :
        #for every possible state, compute the reward
        
        #create a grid of all possible states
        grid = torch.eye(self.state_dim).view(self.state_dim, self.size, self.size)
         
        #compute the reward for each state using the reward function
        rewards = self.reward(grid)
        
        return rewards