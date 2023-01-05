import torch
from gflownet.env import Env
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import radius_graph


from torch_geometric.utils import to_networkx
import networkx as nx

"""This represents a graph of cells that can move around.
"""
class MovingCells(Env) :
    def __init__(self, graph_size, edge_distance=2.) :
        self.graph_size = graph_size
        
        self.num_actions = 7 # create a cell, delete a cell, move a cell up, move a cell right, move a cell down, move a cell left, terminate
        
        #we will need to make sure we can't delete a cell that doesn't exist
        # or move a cell that doesn't exist
        #but also not delete a cell that we just created or move a cell in another direction
        
        self.limit = 50 #the maximum number of actions that can be performed in a row
        self.max_cells = graph_size**2 #the maximum number of cells that can be created
        
        self.state_dim = graph_size ** 2 * 2 #the state is a graph of cells, each cell has two attributes: x and y
        
        self.edge_distance = edge_distance #the maximum distance between two cells to be connected
        
        #TODO: this is a placeholder for the reward function that will be done as a surrogate model
        self.objectives = self.compose_objectives()

    def update(self, s, actions) :
        create, delete, up, right, down, left, terminate = actions == 0, actions == 1, actions == 2, actions == 3, actions == 4, actions == 5, actions == 6
        
        s = s.view(-1, self.graph_size ** 2, 2)
        
        number_of_cells =  torch.sum(~torch.isnan(s), dim=1)[:,0]
        
        #the first attribute is the x coordinate
        #the second attribute is the y coordinate
        
        random_cell = torch.tensor([torch.randint(0, int(cells.item()), (1,)) for cells in number_of_cells[create]], dtype=torch.long)
        random_nan_cell = torch.tensor([torch.randint(self.max_cells - int(cells.item()), self.max_cells, (1,)) for cells in number_of_cells[delete]], dtype=torch.long)
        
        #we can create a new cell by turning one random nan cell into one valid cell
        s[create][:, random_nan_cell] = torch.tensor([torch.randint(0, self.graph_size, (1,)), torch.randint(0, self.graph_size, (1,))], dtype=torch.float).to(s.device)
        
        #we can delete a cell by turning one random cell into one nan cell
        s[delete][:, random_cell] = torch.tensor([float('nan'), float('nan')], dtype=torch.float).to(s.device)
        
        
        s[up][:, random_cell] += torch.tensor([0, 1]).to(s[up].device)
        s[down][:, random_cell] += torch.tensor([0, -1]).to(s[down].device)
        s[right][:, random_cell] += torch.tensor([1, 0]).to(s[right].device)
        s[left][:, random_cell] += torch.tensor([-1, 0]).to(s[left].device)
        
        s = s.view(s.shape[0], -1)
        
        return s
    
    def mask(self, s) :
        s = s.view(-1, self.graph_size ** 2, 2)
        
        mask = torch.ones(s.shape[0], self.num_actions).to(s.device)

        #if number of cells is 0, we can't delete a cell
        #if number of cells is max, we can't create a cell
        
        number_of_cells = torch.sum(~torch.isnan(s), dim=1)[:,0]
        
        mask[:, 1] = number_of_cells > 0
        mask[:, 0] = number_of_cells < self.max_cells
        
        return mask
    
    def terminal_action(self, actions):
        return actions == 6
    
    def terminal_state(self, s, iteration=0):
        s = s.view(-1, self.graph_size ** 2, 2)
        return torch.repeat_interleave(torch.tensor(iteration >= self.limit, dtype=torch.bool), len(s)).to(s.device)
    
    def reward(self, s):
        #there are 4 objectives and we need to be close to all of them
        s = s.view(-1, self.graph_size ** 2, 2)
        rewards  = torch.zeros(len(s)).to(s.device)
        for f in self.objectives :
            rewards  = rewards + torch.tensor([f(si) for si in s]).to(s.device)
            
        return rewards
    
    def compose_objectives(self):
        obj = []
        
        #random objective to have the closest to five cells
        def f1(s) :
            number_of_cells = torch.sum(~torch.isnan(s), dim=0)[0]
            return 1 / (torch.abs(number_of_cells - 5) + 1)
        
        #other objective is that the cells should be connected
        def f2(s) :
            #compute the edge_attribute which is the distance between two cells
            edge_index = radius_graph(s, self.edge_distance)
            edge_attr = torch.norm(s[edge_index[0]] - s[edge_index[1]], dim=1)
            
            #have a reward that is the average distance between cells to be close to 0
            return 1 / ((torch.mean(edge_attr) / self.edge_distance) + 1)
        
        obj.append(f1)
        
        return obj
    
    def plot(self, s, fig_name):
        plt.figure()
        
        s = s.view(-1, self.graph_size ** 2, 2)
        
        s = s[torch.argmax(self.reward(s))]
        
        s = s[~torch.isnan(s).any(dim=1)]
        
        edge_index = radius_graph(s, self.edge_distance)
        
        graph = Data(pos=s, 
                        edge_index=edge_index)
        g = to_networkx(graph, to_undirected=True)
        nx.draw(g)
        
        plt.show()