import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        =====
            state_size (int): Dimensions of each state
            action_size (int): Dimensions of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed) #set the seed for generating random numbers. Returns torch.generator object
        self.fc1 = nn.Linear(state_size, fc1_units) #Take states and pass into fully connected first hidden layer
        self.fc2 = nn.Linear(fc1_units, fc2_units) #Take output of fully connected first hidden layer and pass into second fully connected hiddent layer
        self.fc3 = nn.Linear(fc2_units, action_size) #Take output of second fully connected hidden layer and pass into actions to be taken

    def forward(self, state):
        """
        Forward proppagation
        Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state)) #Apply rectified linear action funcction to fc1 output after passing state into fc1
        x = F.relu(self.fc2(x)) #Apply rectified linear action function to fc2 output after passing into fc1's output
        return self.fc3(x) #Apply rectified linear action function to fc3 output after passing into fc2's output