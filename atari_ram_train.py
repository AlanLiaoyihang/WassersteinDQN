import torch 
from torch.nn import functional as F
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
import logging
import os

#OpenAI gym and cherry packages
import gym
import cherry as ch
from cherry import envs

#my code
from algorithms import DoubleDQN,WassersteinDQN,Curiosity_driven_DQN
from my_Wrapper import stats_wrapper


"""
this is the Atari training for ram input
same structure as classic control but
hidden size is larger, not accesses to cloud 
resources, local computer not able to complete 
a decent experiments, but the function could run
with not good hyperparameters
"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Hyper-parameters settings
DISCOUNT = 0.99
HIDDEN_SIZE = 512
LR = 0.001
MAX_STEPS = 200000
BATCH_SIZE = 512
REPLAY_SIZE = 100000
UPDATE_INTERVAL = 1
TARGET_UPDATE_INTERVAL = 1000
UPDATE_START = 50
SEED = 42

#logger setting

#recorde last 5 episode rewards
EP_INTERVAL = 5

#record last 1000 steps rewards
STEP_INTERVAL = 1000

#record every 2000 steps
RECORD_INTERVAL = 2000
RECORD_EPISODIC = False


#if CUDA isavaliable use gpu
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

#normalizing weights
def weights_init(m):
    """
    initilalize the weights
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)

#create folder
def mkdir(path):
    """
    if path does not exists, creates
    @path: the path you want to save
    """
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print('Successfully create path')
        return True
    else:
        return False

#save stats into csv files
def save_to_csv(steprewards,stepci, eprewards, epci, name, env_name, path):
    """
    this function is to save the rewards into a data files, given a record
    of the training with path
    @steprewards: rewards of past STEP_INTERVAL steps
    @stepci: variance of past STEP_INTERVAL steps rewarrds
    @eprewards: rewards of past EP_INTERVAL steps
    @epci: variance of past EP_INTERVAL steps rewarrds
    """
    folder_path =  '{}/{}'.format(path,env_name)
    mkdir(folder_path)

    data = np.array([steprewards, stepci, eprewards, epci]).T
    df2 = pd.DataFrame(data, columns = ['Stepsrewards','Stepsci','Episoderewards','Episodeci'])
    df2.to_csv(folder_path + '/{}_DQN_rewards'.format(name))



def RAM_train(agent_func,name,env_name,figure_path, logs_path, file_path, eps = 0, action_discretization = 10):
    
    def set_seed(SEED):
        """
        setting seed of environments
        """
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        env.seed(SEED)

    #clean the existing log files
    logs_folder_path =  '{}/{}'.format(logs_path,env_name)
    mkdir(logs_folder_path)

    logs_directories = logs_folder_path + '/{}_DQN.log'.format(name)
    if os.path.exists(logs_directories):
        os.remove(logs_directories)
        print("delete ...." )

    #loading the environments with normalizer
    env = ch.envs.openai_atari_wrapper.make_atari(env_name)

    #set seed
    set_seed(SEED)

    #wrapping to accpet torch Tensor
    env = envs.Torch(env)

    #logger wrapper
    env = stats_wrapper(env, record_interval= RECORD_INTERVAL,step_interval= STEP_INTERVAL, episode_interval= EP_INTERVAL, record_episodic= RECORD_EPISODIC)
    #setting logger output files direction
    file_handler = logging.FileHandler(logs_directories)
    file_handler.setLevel(logging.DEBUG)
    env.logger.addHandler(file_handler)

    state_size = env.observation_space.shape[0]
    action_space = env.action_space

    #check whether continuous or discrete action domain
    if isinstance(action_space,gym.spaces.Box):
        """
        if box action space means the action space is continuous
        """
        action_size = action_discretization
        actions = torch.linspace(-action_space.low[0],action_space.high[0],action_discretization)
    
        def convert_discrete_to_continuous_action(action):
            return actions[action.long()].to(dtype=torch.float32)
        
        env = envs.ActionLambda(env,convert_discrete_to_continuous_action)
    else:
        action_size = action_space.n

    #runner wrapper
    env = envs.Runner(env)

    #adding replay buffer
    replay = ch.ExperienceReplay()

    #initialise agent
    agent = agent_func(state_size,HIDDEN_SIZE, action_size,LR,DISCOUNT, eps = eps).to(device)
    agent.apply(weights_init)
    
    #training 
    episodes = 0
    reward = 0
    rewards = []
    for steps in range(MAX_STEPS):
        with torch.no_grad():
            if episodes < UPDATE_START:
                replay += env.run(agent.random_action, steps = 1).to(device)
            else:
                replay += env.run(agent.choose_action, steps = 1).to(device)

        if len(replay) > REPLAY_SIZE:
            replay = replay[-REPLAY_SIZE:]

        steps += 1

        reward += replay[-1].reward.view(1)

        if replay[-1].done.view(1):
            episodes += 1
            rewards.append(reward)
            reward = 0


        if episodes > UPDATE_START and steps % UPDATE_INTERVAL == 0:
            batch = replay.sample(BATCH_SIZE)

            agent.train(batch)

        if episodes > UPDATE_START and steps % TARGET_UPDATE_INTERVAL == 0:
            # Update target network
            agent.update_target_network()

    # step_stats = env.all_rewards
    agent.save_weights("./weights/{}_DQN_{}".format(name,env_name))

    save_to_csv(env.interval_mean_rewards,env.ci,env.episodic_mean_rewards,env.epci,name = name,env_name = env_name,path = file_path)

    env.close()
    env.logger.removeHandler(file_handler)
    return env

def plot_ci(rewards,ci,ax,interval = 1000):
    """
    this plots the rewards and its confidence interval against the number of episodes
    """
    x = range(0,len(rewards),interval)
    index = np.arange(0,len(rewards),interval)
    upper_quantile = rewards + ci
    lower_quantile = rewards - ci

    ax.plot(index,rewards[x],'*-')
    ax.fill_between(index,lower_quantile[x],upper_quantile[x],alpha = .1)



if __name__ == '__main__':
    env_name = ['Breakout-ramNoFrameskip-v4','Pong-ramNoFrameskip-v4']
    algo = [WassersteinDQN,DoubleDQN,DoubleDQN, Curiosity_driven_DQN]
    name = ['Wasserstein','Double_greedy','Double_egreedy','ICM']
    epsilon = [0, 0, 0.05, 0]
    rewards_steps = []
    ci_steps = []
    rewards_episodic = []
    ci_episodic = []

    # train all environments
    # for i in range(len(env_name)):
    #     for j in range(len(algo)):
    #         RAM_train(algo[j],name[j],env_name[i],'figures', 'Ep_train_log', 'EP_train_datafiles',eps = epsilon[j])
    i = 1
    j = 0
    RAM_train(algo[j],name[j],env_name[i],'figures', 'Ep_train_log', 'EP_train_datafiles',eps = epsilon[j])
