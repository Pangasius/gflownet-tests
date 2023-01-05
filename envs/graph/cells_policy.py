import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv

from torch_geometric.nn import radius_graph
from torch_geometric.data import Data

class ForwardPolicy(torch.nn.Module) :
    def __init__(self, num_actions, graph_size, edge_distance, hidden_dim) :
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.out_channels = 32
        self.heads = 4
        self.num_actions = num_actions
        self.graph_size = graph_size
        
        self.edge_distance = edge_distance
        
        self.t1 = TransformerConv(2, self.out_channels, heads=self.heads, dropout=0.1, edge_dim=1)
        
        #this is a classification problem, so we need to output a probability distribution
        self.conv = torch.nn.Conv2d(self.out_channels * self.heads, 2, kernel_size=2)
        
        self.dense1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dense2 = torch.nn.Linear(self.hidden_dim, self.num_actions)
        
    def forward(self, data):
        data = data.view(-1, self.graph_size ** 2, 2)
                
        batch = torch.arange(data.shape[0], dtype=torch.long).to(data.device)
        batch = batch.repeat_interleave(data.shape[1])
        
        batch_size = data.shape[0]
        
        data = data.view(-1, 2)
        
        final = torch.zeros(batch_size, self.num_actions).to(data.device)
        
        #TODO: optimize this to work in batch
        for i in range(batch_size):
            data_bis = data[batch == i].to(data.device)
            
            data_bis[torch.isnan(data_bis)] = -10
        
            edge_index = radius_graph(data_bis.to("cpu"), r=self.edge_distance, loop=False, max_num_neighbors=4).to(data_bis.device)
            edge_attr = torch.norm(data_bis[edge_index[0]] - data_bis[edge_index[1]], dim=1).view(-1, 1).to(data_bis.device)
            
            data_bis = F.relu(self.t1(data_bis, edge_index, edge_attr).to(data_bis.device))
            data_bis = self.conv(data_bis.view(1, self.out_channels * self.heads, self.graph_size, self.graph_size)).view(-1)
            data_bis = F.relu(self.dense1(data_bis))
            data_bis = self.dense2(data_bis)
            data_bis = F.softmax(data_bis, dim=0)
            
            final[i] = data_bis
        
        return final.view(-1, self.num_actions)
    
class BackwardPolicy:
    def __init__(self, num_actions, graph_size):
        super().__init__()
        self.num_actions = num_actions
        self.graph_size = graph_size
    
    def __call__(self, s):
        s = s.view(-1, self.graph_size ** 2, 2)
        
        probs = torch.ones(len(s), self.num_actions).to(s.device)
        
        """
        create, delete, up, right, down, left, terminate = actions == 0, actions == 1, actions == 2, actions == 3, actions == 4, actions == 5, actions == 6
        """
        
        number_of_cells = torch.sum(~torch.isnan(s), dim=1)[:,0]
        
        #can't backprop create if there are no nodes
        probs[number_of_cells > 0, 0] = 0
        
        #can't backprop delete if there are self.graph_size**2 nodes
        probs[number_of_cells == self.graph_size**2, 1] = 0
        
        return probs
