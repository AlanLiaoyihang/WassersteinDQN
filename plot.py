"""
this is the code for generating plots from the saved record of the training
"""


from os import read
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

env_name_list = ['Pendulum-v0','CartPole-v1','MountainCar-v0','Acrobot-v1']
name = ['Wasserstein','Double_greedy','Double_egreedy','ICM']
figure_path, logs_path, file_path = 'figures', 'Ep_train_log', 'EP_train_datafiles'
index = ['Stepsrewards','Stepsci','Episoderewards','Episodeci']


def read(path, index, window = 1):
    """
    rolling mean for plottings
    """
    a = pd.read_csv(path)
    return a[index[0]].rolling(window=window).mean(),a[index[1]].rolling(window=window).mean(),a[index[2]].rolling(window=window).mean(),a[index[3]].rolling(window=window).mean()

def plot_ci(rewards,ci,ax,interval = 1000, ep_interval = 5):
    """
    rewards: array of rewards
    ci: std of rewards
    interval: plot interval
    """
    x = range(0,len(rewards),interval)
    index = np.arange(0,len(rewards),interval)*ep_interval
    upper_quantile = rewards + ci
    lower_quantile = rewards - ci

    ax.plot(index,rewards[x],'*-')
    ax.fill_between(index,lower_quantile[x],upper_quantile[x],alpha = .3)


if __name__ == "__main__":
    for env_name in env_name_list:
        rewards_steps,ci_steps,rewards_episodic,ci_episodic = [], [], [], []
        for i in range(len(name)):
            folder_path =  '{}/{}'.format(file_path,env_name)
            file_directories = folder_path + '/{}_DQN_rewards'.format(name[i])
            # a = pd.read_csv(file_directories)
            rs,rci,epsstats,epci = read(file_directories,index, window = 4)
            rewards_steps.append(rs)
            ci_steps.append(rci)
            rewards_episodic.append(epsstats)
            ci_episodic.append(epci)
        fig,ax = plt.subplots()
        for i in range(len(name)):
            plot_ci(np.array(rewards_steps[i]),np.array(ci_steps[i]),ax, interval = 8)

        ax.set_xlabel('episodes')
        ax.set_ylabel('avg_steps_reward')
        ax.set_title('{}_StepStats_Train'.format(env_name))
        ax.legend(name, loc = 'upper left')
        fig.savefig('figure/{}_steps.jpg'.format(env_name))

        fig1,ax1 = plt.subplots()
        for i in range(len(name)):
            plot_ci(np.array(rewards_episodic[i]),np.array(ci_episodic[i]),ax1, interval = 8)

        ax1.set_xlabel('episodes')
        ax1.set_ylabel('avg_episodes_reward')
        ax1.set_title('{}_EpStats_Train'.format(env_name))
        ax1.legend(name, loc = 'upper left')
        fig1.savefig('figure/{}_episodes.jpg'.format(env_name))