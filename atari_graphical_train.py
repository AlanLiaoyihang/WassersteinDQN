import torch 
from torch.nn import functional as F
import numpy as np
import random
import gym
import cherry as ch
from cherry import envs
from algorithms import Atari_WassersteinDQN,Atari_DoubleDQN
import logging
import os
from my_Wrapper import stats_wrapper
from matplotlib import pyplot as plt
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Hyper-parameters settings
DISCOUNT = 0.99
# HIDDEN_SIZE = 128
LR = 0.001
MAX_STEPS = 500000
BATCH_SIZE = 256
REPLAY_SIZE = 20000
UPDATE_INTERVAL = 1
TARGET_UPDATE_INTERVAL = 100
UPDATE_START = 1000
SEED = 42

EP_INTERVAL = 5
STEP_INTERVAL = 2000
RECORD_INTERVAL = 5
RECORD_EPISODIC = True


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


def mkdir(path):

    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print('Successfully create path')
        return True
    else:
        return False

def save_to_csv(steprewards,stepci, eprewards, epci, name, env_name, path):
    """
    this function is to save the rewards into a data files, given a record
    of the training
    """
    folder_path =  '{}/{}'.format(path,env_name)
    mkdir(folder_path)

    data = np.array([steprewards, stepci, eprewards, epci]).T
    df2 = pd.DataFrame(data, columns = ['Stepsrewards','Stepsci','Episoderewards','Episodeci'])
    df2.to_csv(folder_path + '/{}_DQN_rewards'.format(name))



def Atari_train(agent_func,name,env_name,figure_path, logs_path, file_path, eps = 0, action_discretization = 10):
    def set_seed(SEED):
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

    #loading the environments
    env = gym.make(env_name)

    #Atari Wrapper
    env = ch.envs.OpenAIAtari(env)

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

    #state_size = env.observation_space.shape[0]
    action_space = env.action_space

    #check whether continuous or discrete action domain
    if isinstance(action_space,gym.spaces.Box):
        """
        if box action space means the action space is continuous
        """
        action_size = action_discretization
        actions = torch.linspace(-action_space.low[0],action_space.high[0],action_discretization)
    
        def convert_discrete_to_continuous_action(action):
            return actions[action.long()]
        
        env = envs.ActionLambda(env,convert_discrete_to_continuous_action)
    else:
        action_size = action_space.n
    print(action_size)

    #runner wrapper
    env = envs.Runner(env)

    #adding replay buffer
    replay = ch.ExperienceReplay()

    #initialise agent
    agent = agent_func(action_size = action_size,learning_rate = LR,discount = DISCOUNT, eps = eps).to(device)
    agent.apply(weights_init)
    #training 
    episodes = 0
    steps = 0
    while steps < MAX_STEPS:
        with torch.no_grad():
            if steps < UPDATE_START:
                replay += env.run(agent.random_action, steps = 1).to(device)
            else:
                replay += env.run(agent.choose_action, steps = 1).to(device)
        
        #if seeing the game processing
        env.render()

        steps += 1

        if len(replay) > REPLAY_SIZE:
            replay = replay[-REPLAY_SIZE:]

        if replay[-1].done.view(1):
            episodes += 1

        if steps > UPDATE_START and steps % UPDATE_INTERVAL == 0:
            batch = replay.sample(BATCH_SIZE)
            agent.train(batch)

        if steps % TARGET_UPDATE_INTERVAL == 0:
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
    env_name = ['BreakoutNoFrameskip-v0']
    algo = [Atari_WassersteinDQN,Atari_DoubleDQN,Atari_DoubleDQN]
    name = ['Wasserstein','Double_greedy','Double_egreedy']
    epsilon = [0, 0, 0.05, 0]
    rewards_steps = []
    ci_steps = []
    rewards_episodic = []
    ci_episodic = []

    #train all environments
    # for i in range(len(env_name)):
    #     for j in range(len(algo)):
    #         classic_control_train(algo[j],name[j],env_name[i],'figures', 'Ep_train_log', 'EP_train_datafiles',eps = epsilon[j])
    i = 0
    j = 0
    Atari_train(algo[j],name[j],env_name[i],'figures', 'Ep_train_log', 'EP_train_datafiles',eps = epsilon[j])