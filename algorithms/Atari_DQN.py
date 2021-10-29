#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   Yihang Liao
@File    :   DQN.py
@LastEditors    :   2021/08/23
@Description    :   The base algorithms used in final projects for graphical input
'''


import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import numpy as np
import cherry as ch


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConvNet(nn.Module):
    """
    The ConvNet used for images input
    """
    def __init__(self, input_size=4, action_size = 4, hidden_size=64*7*7):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.conv2d1 = nn.Conv2d(input_size, 32, 8, stride=4, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.conv2d3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU()
        self.flat = Flatten()
        self.fc1 = nn.Linear(hidden_size, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.out = nn.Linear(256, action_size)

    def forward(self,state):
        state =  state.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        out = self.conv2d1(state)
        out = self.relu1(out)
        out = self.conv2d2(out)
        out = self.relu2(out)
        out = self.conv2d3(out)
        out = self.relu3(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.relu5(out)
        out = self.out(out)
        
        return out


class Atari_DoubleDQN(nn.Module):
    """
    Double Q learning algorithm, Double Q learning make adjustment to 
    calculate the expected

    @input_size: specify the state size
    @hidden_size: hidden size of the neural network
    @action_size: number of actions
    @learning_rate: optimizer learning rate
    @discount: discount factor
    """
    def __init__(self, input_size=4, action_size = 6, hidden_size=64*7*7 ,discount = 0.99, eps = 0, learning_rate = 0.001):
        super().__init__()
        #agents
        self.agent = ConvNet(input_size,action_size, hidden_size)
        self.target_network = self.create_target_network()
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr = learning_rate)
        self.scheduler =  torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.99)
        self.discount = discount
        #paramters
        self.learning_rate = learning_rate
        self.eps = eps
        self.action_size = action_size
        self.egreedy = ch.nn.EpsilonGreedy(eps)

    
    def create_target_network(self):
        target_network = copy.deepcopy(self.agent)
        for param in target_network.parameters():
            param.requires_grad = False
        return target_network

    def choose_action(self,state):
        values = self.agent(state.to(device))
        actions = values.max(dim=1,keepdim = True)[1]
        sampled = torch.distributions.Categorical(probs=torch.ones_like(values)).sample().to(device)
        probs = torch.ones(values.size(0), 1) - self.eps
        b = torch.distributions.Bernoulli(probs=probs).sample().long().to(device)
        res = actions * b + (1 - b) * sampled
        return res
    
    def random_action(self,state):
        return torch.randint(self.action_size-1,size = (1,))

    def update_target_network(self):
        self.target_network = self.create_target_network()
    
    def train(self,batch):
        target_values = self.target_network(batch.next_state()).max(dim=1, keepdim=True)[0]
        target_values = batch.reward() + self.discount * (1 - batch.done()) * target_values

        #updating network
        pred_values = self.agent(batch.state()).gather(1, batch.action())
        value_loss = F.mse_loss(pred_values, target_values)
        
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        # self.scheduler.step()
    
    def save_weights(self,path):
        torch.save(self.agent.state_dict(), path)
    
    def load_weights(self,path):
        self.agent.load_state_dict(torch.load(path))
        self.agent.eval()

        

class Atari_WassersteinDQN(Atari_DoubleDQN):
    """
    Wasserstein DQN algorithm

    @input_size: specify the state size
    @hidden_size: hidden size of the neural network
    @action_size: number of actions
    @learning_rate: optimizer learning rate
    @discount: discount factor
    """
    def __init__(self, input_size=4, action_size = 6, hidden_size=64*7*7 ,discount = 0.99, eps = 0, learning_rate = 0.001):
        super().__init__(input_size=input_size, action_size = action_size, hidden_size=hidden_size ,discount = discount, eps = eps, learning_rate = learning_rate)
        self.action_size = action_size

    def compute_prob_max(self,Q):
        score = torch.as_tensor((Q.T[:,:,None,None] >= Q.T),dtype=torch.int)
        prob = score.sum(axis=3).prod(axis = 2).sum(axis = 1)
        return torch.as_tensor(prob,dtype = torch.float32)/prob.sum()
    
    def V_posterior(self,batch):
        Q_next = self.agent(batch.next_state())
        target_Q =self.target_network(batch.next_state())
        
        Q_next = Q_next.sort(dim = 0)[0]
        prob_tar = self.compute_prob_max(Q_next).reshape(-1,1)

        targets_V = torch.mm(target_Q, prob_tar)

        Q = self.agent(batch.state()).gather(1,batch.action())


        return targets_V, Q

    def train(self,batch): 
        targets_V, V = self.V_posterior(batch)
        target_values = batch.reward() + self.discount*(1-batch.done())*targets_V
        pred_values = V

        value_loss = F.mse_loss(pred_values, target_values).clip(-1,1)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        # self.scheduler.step()

#imcomplete for conv Curiosity dqn.