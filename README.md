
# Yihang Liao ACSE-9 individual project
Currently implement Double DQN, WassersteinDQN, Curiosity driven DQN in OpenAI gym classic control environment. The train on Atari games with ram input could be run but have no time to complete the training to obtain results. The train on Atari game using RGB image input is not complete, only work for Double DQN

## Project summary
This project includes the study of Wasserstein Barycenter and its application in reinforcement learning.
Wasserstein Barycenter is a natural way to combine uncertainty of value function in DQN and backpropagates
to agent's network, which enables agent to balancing the trade off between exploration and eploitations

## Dependency:
* pytorch 1.9.0
* gym
* cherry-rl
* matplotlib
* pandas

* CUDA 11.0(not compulsory)
* Cudnn 0.9.0(not compulsory)

## Final report link:

* [project report](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-yl27218/blob/master/acse-yl27218_Finalreport.pdf)


## Usage:
In  order to run my training of classic control, statisfying the required dependency, run

``` python
python classic_train.py
```
After the training, the stastistics could be plot via the code:

``` python
python plot.py
```

The hyperparameters could be changed at the top of the classic_train.py
The training of Atari game is imcompleted, but no time to train through with
enough episodes

``` python
python Atari_ram_train.py
```

The convolutional network is imcompleted, it is work for future

``` python
python Atari_graphical_train.py
```

## Folder 
*   datafile folder stores the stats into .csv format
[datafiles_path](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-yl27218/tree/master/EP_train_datafiles)

*   log folder stores the stats into .csv format
[Log_path](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-yl27218/tree/master/Ep_train_log)

*   weights folder stores the weights of different algorithms in environments
[weights](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-yl27218/tree/master/weights)

## Main results of classic control problems
* plots as shown:

![image](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-yl27218/blob/master/figure/Acrobot-v1_episodes.jpg "bibtex_simulation")
![image](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-yl27218/blob/master/figure/CartPole-v1_episodes.jpg "bibtex_simulation")
![image](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-yl27218/blob/master/figure/MountainCar-v0_episodes.jpg "bibtex_simulation")
![image](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-yl27218/blob/master/figure/Pendulum-v0_episodes.jpg "bibtex_simulation")


