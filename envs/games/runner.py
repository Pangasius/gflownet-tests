import torch
from gflownet.env import Env
import matplotlib.pyplot as plt

"""This environment roughly implements a runner game like the chrome dinosaur game.
"""
class Runner(Env) :
    def __init__(self, length):
        self.max_length = length
        self.num_actions = 3 # jump, duck, nothing
        
        #the state is simply the position of the runner for the score,
        #weither the runner is ducking or not, weither the runner is jumping or not
        #depending in the position, obstacles can appear
        #at length == max_length, the game is over
        
        #the following are the modulo at which obstacles appear
        self.obstacle_jump = 3
        self.obstacle_duck = 4
        
        #state = position, ducking, jumping, dead
        self.state_dim = 4
        
    def update(self, s, actions):
        jump, duck, nothing = actions == 0, actions == 1, actions == 2
        
        #update the position
        s[:, 0] = s[:, 0] + 1
        
        reset = torch.zeros_like(s[:, 1])
        
        #update the ducking
        s[:, 1] = reset + duck.float()
        
        #update the jumping
        s[:, 2] = reset + jump.float()

        #if the runner is on an obstacle, they die
        death_on_jump = torch.logical_and(s[:, 0] % self.obstacle_jump == 0, torch.logical_not(s[:, 2]))
        death_on_duck = torch.logical_and(s[:, 0] % self.obstacle_duck == 0, torch.logical_not(s[:, 1]))
        
        #if both obstacle, then they are deleted
        death_prevented = torch.logical_not(torch.logical_and(s[:, 0] % self.obstacle_jump == 0, s[:, 0] % self.obstacle_duck == 0))
        
        s[:, 3] = torch.logical_or(s[:, 3], torch.logical_and(torch.logical_or(death_on_jump, death_on_duck), death_prevented))

        return s.float()
    
    def mask(self, s):
        mask = torch.ones(len(s), self.num_actions).to(s.device)
        return mask
    
    def reward(self, s):
        #the reward is the position of the runner if he is alive
        return s[:, 0]
    
    def terminal_state(self, s, iteration=0):
        return torch.logical_or(s[:, 0] >= self.max_length, s[:, 3] == 1)
    
    def terminal_action(self, actions):
        return torch.zeros(len(actions)).to(actions.device)
    
    def plot(self, s, fig_name):
        plt.figure()
        
        rewards = torch.tensor(self.reward(s))
        
        print("Mean reward: ", torch.mean(rewards).item())
        print("Max reward: ", torch.max(rewards).item())
        
        #make an histogram of the rewards
        plt.hist(rewards, bins=int(torch.max(rewards).item()))
        plt.title("Rewards")
        
        plt.savefig(fig_name)
        
        plt.show()
        