# AI for Self Driving Car

# Importing the libraries

import numpy as np #library that allows us to play amd work with arrays
import random #to take some random samples from different batch
import os #load the model and save the brain 
import torch #implementing neural network with pytorch
import torch.nn as nn #nn modules contains all the tools to implement neural network
import torch.nn.functional as F #different functions we use when implementing neural network
import torch.optim as optim #optimizers
import torch.autograd as autograd 
from torch.autograd import Variable #convert tensor into a variable containing tensor and gradient

# Creating the architecture of the Neural Network

# --> init function(defines the variables attached to the object)
# --> input layer(5 neurons as there are five dim of encoded vector state)
# --> hidden layer
# --> output layer(possible actions)

# --> forward function(will activate the neurons in the neural n/w)
# --> Rectifier Activation Function as we are dealing with non-linear problems
# --> forward function will return the Q-values that are the output of the neural n/w

class Network(nn.Module): #inheriting from the Module parent class

    def __init__(self, input_size, nb_action): #number of input neurons(2nd arg), o/p neurons or actions(3rd arg)
        super(Network, self).__init__() #to use all methods of nn Module that we inherited
        self.input_size = input_size
        self.nb_action = nb_action
        #2 full connections b/w the neurons of the i/p layer to the neurons of the o/p layer
        #here we choose 30 neurons as o/p we can choose it to any no. depending on us
        self.fc1 = nn.Linear(input_size, 30) #connection b/w i/p layer to hidden layer
        self.fc2 = nn.Linear(30, nb_action) #connection b/w hidden layer to o/p layer

    def forward(self, state): #state is the input to the neural n/w
        # Relu is the rectifier function that will activate the hidden neurons
        # we will take our first full connection FC1 which will apply to our i/p neurons to go from the i/p neurons to the hidden neurons
        x = F.relu(self.fc1(state)) #hidden neurons
        q_values = self.fc2(x)
        return q_values
    
    
#Implementing Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity): # capacity stores the previous exeriences(100)
        # self.capacity is the name of variable attached to the object, 
        # capacity is argument we will input when creating object of replay memory class
        self.capacity = capacity
        self.memory = [] #will contain last 100 events
        
    # push function will append new transition/experience in the memory
    # make sure memory has always 100 transitions
    # event is a tuple of four elements: last_state(st), new_state(st+1), last_action(at), last_reward(rt)
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    # sample function will take random samples from the memory
    # batch_size will have fixed number of samples we want at a time
    def sample(self, batch_size):
        # zip * reshapes the list, by using it we are grouping state, rewards and actions together
        samples = zip(*random.sample(self.memory, batch_size))
        # map function will do mapping from samples to torch variables
        # each batch in sample like batch of actions, 
        # we have to concatenate it with respect to first dimension which corresponds to state
        # so that everything is well aligned like for each row the state, reward and action corresponds to the same time
        # the  lambda fn will take the samples concatenate them wrt to 1st dim 
        # and then eventually we convert these tensors into some Torch vaariable that contains both tensor and gradient
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
#Implementing Deep Q Learning 

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):# gamma is delay coefficient, parameter of equation
        self.gamma = gamma
        self.reward_window = []# mean of reward in reward_window that will slide over time, mean of last 100 rewards increasing with time
        self.model = Network(input_size, nb_action) # neural n/w of DQ model
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters, lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state): # i/p of neural network are states
        probs = F.softmax(self.model(state))
