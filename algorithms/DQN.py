#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   Yihang Liao
@File    :   DQN.py
@LastEditors    :   2021/07/01
@Description    :   The base algorithms used in final projects for non graphical input
'''


import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import numpy as np
import cherry as ch


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Agent(nn.Module):
    """
    This is the deep Q network agent base

    @input_size: specify the state size
    @hidden_size: hidden size of the neural network
    @action_size: number of actions
    """
    def __init__(self, input_size, hidden_size, action_size):
        super(Agent,self).__init__()
        self.action_size = action_size

        self.fc1 = nn.Linear(input_size,hidden_size)
        #activation function
        self.af1 =  nn.ReLU()

        #second fully connected layers
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.af2 = nn.ReLU()

        #output
        self.last = nn.Linear(hidden_size,action_size)

    def forward(self, state):
        output = self.fc1(state)
        output = self.af1(output)
        output = self.fc2(output)
        output = self.af2(output)
        output = self.last(output)

        #estimates Q values
        return output



class DoubleDQN(nn.Module):
    """
    Double Q learning algorithm, Double Q learning make adjustment to 
    calculate the expected

    @input_size: specify the state size
    @hidden_size: hidden size of the neural network
    @action_size: number of actions
    @learning_rate: optimizer learning rate
    @discount: discount factor
    @eps: epsilon
    """
    def __init__(self, input_size, hidden_size, action_size, learning_rate,discount,eps = 0):
        super().__init__()
        #agents
        self.agent = Agent(input_size,hidden_size,action_size)
        self.target_network = self.create_target_network()
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr = learning_rate)

        #scheduler could decrease the learning rate
        self.scheduler =  torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.99)
   
        #paramters
        self.learning_rate = learning_rate
        self.eps = eps
        self.action_size = action_size
        self.discount = discount

    #create the target network
    def create_target_network(self):
        target_network = copy.deepcopy(self.agent)
        for param in target_network.parameters():
            param.requires_grad = False
        return target_network

    #choose action e-greedily, if not e-greedy, eps = 0 
    def choose_action(self,state):
        values = self.agent(state.to(device))
        actions = values.max(dim=1,keepdim = True)[1]
        sampled = torch.distributions.Categorical(probs=torch.ones_like(values)).sample().to(device)
        probs = torch.ones(values.size(0), 1) - self.eps
        b = torch.distributions.Bernoulli(probs=probs).sample().long().to(device)
        res = actions * b + (1 - b) * sampled
        return res
    
    #random action selection
    def random_action(self,state):
        return torch.randint(self.action_size-1,size = (1,))

    #update target network
    def update_target_network(self):
        self.target_network = self.create_target_network()
    
    #training
    def train(self,batch):
        #compute target values of next state
        target_values = self.target_network(batch.next_state()).max(dim=1, keepdim=True)[0]
        
        #expected return calculation
        target_values = batch.reward() + self.discount * (1 - batch.done()) * target_values

       
        pred_values = self.agent(batch.state()).gather(1, batch.action())
        value_loss = F.mse_loss(pred_values, target_values)
        
        #updating network
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        # if decrease learning rate
        # self.scheduler.step()
    
    #save the weights
    def save_weights(self,path):
        torch.save(self.agent.state_dict(), path)
    
    #load the weights
    def load_weights(self,path):
        self.agent.load_state_dict(torch.load(path))
        self.agent.eval()

        

class WassersteinDQN(DoubleDQN):
    """
    Wasserstein DQN algorithm

    @input_size: specify the state size
    @hidden_size: hidden size of the neural network
    @action_size: number of actions
    @learning_rate: optimizer learning rate
    @discount: discount factor
    """
    def __init__(self, input_size, hidden_size, action_size, learning_rate,discount,eps = 0):
        super().__init__(input_size, hidden_size, action_size, learning_rate,discount,eps)
        #specify action size
        self.action_size = action_size

    #compute the probability distributions of actions given the batch of action-values
    def compute_prob_max(self,Q):
        #generate dirac delta
        score = torch.as_tensor((Q.T[:,:,None,None] >= Q.T),dtype=torch.int)

        #probability
        prob = score.sum(axis=3).prod(axis = 2).sum(axis = 1)
        return torch.as_tensor(prob,dtype = torch.float32)/prob.sum()
    
    #calculate barycenter of target_Q
    def V_posterior(self,batch):
        Q_next = self.agent(batch.next_state())
        target_Q =self.target_network(batch.next_state())
        
        Q_next = Q_next.sort(dim = 0)[0]
        prob_tar = self.compute_prob_max(Q_next).reshape(-1,1)

        targets_V = torch.mm(target_Q, prob_tar)

        Q = self.agent(batch.state()).gather(1,batch.action())


        return targets_V, Q

    #train
    def train(self,batch): 
        targets_V, V = self.V_posterior(batch)
        target_values = batch.reward() + self.discount*(1-batch.done())*targets_V
        pred_values = V

        value_loss = F.mse_loss(pred_values, target_values)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        # self.scheduler.step()

class dynamic_model(nn.Module):
    """
    implementation of ICM model, which generates intrinsic rewards
    @input_dim: dimension of input
    @hidden_size: hidden size of network
    @action_size: action size of agent
    """
    def __init__(self,input_dim,hidden_size, action_dim, phi_dim = 32):
        super(dynamic_model,self).__init__()

        #encoder
        self.encoder = nn.Sequential(
                                            nn.Linear(input_dim, hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, phi_dim),
                                            nn.ReLU()
                                        )
        #forward model
        self.forward_model = nn.Sequential(
                                            nn.Linear(phi_dim + 1, hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, phi_dim),
                                            nn.ReLU()
                                        )
        #inverse model
        self.inverse_model = nn.Sequential(
                                            nn.Linear(phi_dim*2, hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, action_dim),
                                            nn.ReLU()
                                        )
    #phi state by encoder
    def phi_state(self, state):
        return self.encoder(state)
    
    #s1 hat by forward model
    def s1_hat(self, s, action):
        phi_s = self.phi_state(s).detach()
        in_feature = torch.cat([phi_s,action],dim = -1)
        s1_hat = self.forward_model(in_feature)
        return s1_hat
    
    #intrinsic reward
    def intrinsic_reward(self, state, action, next_state):
        s1 = self.encoder(next_state)
        s1_hat = self.s1_hat(state, action)

        return (s1 - s1_hat).norm(dim = 1, keepdim = True)
    
    #inverse model
    def forward(self,state, next_state):
        s = self.phi_state(state)
        s1 = self.phi_state(next_state)        
        in_feature = torch.cat([s,s1],dim = -1)
        pred_action = self.inverse_model(in_feature)
        return pred_action

class Curiosity_driven_DQN(DoubleDQN):
    """
    Curiosity driven DQN Network

    @input_size: specify the state size
    @hidden_size: hidden size of the neural network
    @action_size: number of actions
    @learning_rate: optimizer learning rate
    @discount: discount factor
    """
    def __init__(self, input_size, hidden_size, action_size, learning_rate,discount, intrinsic_scaling = 0.5, eps = 0):
        super(Curiosity_driven_DQN,self).__init__(input_size, hidden_size, action_size, learning_rate,discount,eps = eps)
        
        #create dynamic model and specify optimizer
        self.dynamic_model = dynamic_model(input_size ,hidden_size, 1 , phi_dim= input_size)
        #self.target_dynamic = self.create_target_dynamic()
        self.inverse_optimizer = torch.optim.Adam(self.dynamic_model.inverse_model.parameters(), lr = learning_rate)
        self.forward_optimizer = torch.optim.Adam(self.dynamic_model.forward_model.parameters(), lr = learning_rate)

        #could decrease learning rate
        self.inverse_scheduler = torch.optim.lr_scheduler.StepLR(self.inverse_optimizer, step_size=2000, gamma=0.99)
        self.forward_scheduler =  torch.optim.lr_scheduler.StepLR(self.forward_optimizer, step_size=2000, gamma=0.99)
        self.intrinsic_scaling = intrinsic_scaling

    # #create target network of environment dynamic
    # def create_target_dynamic(self):
    #     target_network = copy.deepcopy(self.dynamic_model)
    #     for param in target_network.parameters():
    #         param.requires_grad = False
    #     return target_network

    #create target network learner
    def update_target_network(self):
        self.target_network = self.create_target_network()
        #self.target_dynamic = self.create_target_dynamic()
    
    def train(self,batch):
        #target values
        target_values = self.target_network(batch.next_state()).max(dim=1, keepdim=True)[0]

        #intrinsic reward
        intrinsic_reward = self.dynamic_model.intrinsic_reward(batch.state(), batch.action(),batch.next_state()).detach()

        #calculate expected return without intrinsic reward
        pred_values = self.agent(batch.state())
        dynamic_pred = self.dynamic_model(batch.state(),batch.next_state())
        expected_return = batch.reward() + self.discount * ((1 - batch.done()) * target_values) 
        
        #s1 and s1 hat
        s1 = self.dynamic_model.encoder(batch.next_state())
        s1_hat = self.dynamic_model.s1_hat(batch.state(), batch.action())


        #dynamic_pred specifies the inverse action value
        target_values = expected_return + self.intrinsic_scaling*intrinsic_reward

        #loss calculation
        value_loss = F.mse_loss(pred_values.gather(1, batch.action()), target_values)
        dynamic_loss = F.mse_loss(batch.reward(), dynamic_pred)
        forward_model_loss = F.mse_loss(s1.detach(),s1_hat)


        #updating model
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        self.inverse_optimizer.zero_grad()
        dynamic_loss.backward()
        self.inverse_optimizer.step()

        self.forward_optimizer.zero_grad()
        forward_model_loss.backward()
        self.forward_optimizer.step()

        #if decaying learning rate
        # self.inverse_scheduler.step()
        # self.forward_scheduler.step()
        # self.scheduler.step()