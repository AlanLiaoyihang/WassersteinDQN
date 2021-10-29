"""
This is my version of Logger which includes collecting the rewards and the standard deviation
of rewards with episodes period and steps periods, modified version of cherry-rl
"""
from cherry.envs.base import Wrapper
from statistics import mean, pstdev
import math
import numpy as np

import cherry as ch


"""
Inspired from cherry-rl

https://github.com/learnables/cherry

"""

class stats_wrapper(ch.envs.Logger):
    """
    This wrapper is to give the training log and record stats, based on the cherry.envs.logger with 
    some modifications
    @env: the environment to wrap
    @record_interval: frequency generating logs and record interval
    @step_interval: frequency record rewards, by steps
    @episode_interval: record rewards, by episodes
    """

    def __init__(self, env,record_interval, step_interval=1000, episode_interval= 5, title=None, logger=None, record_episodic = False):
        super(stats_wrapper, self).__init__(env)
        self.ci = []
        self.epci = []
        self.interval_mean_rewards = []
        self.episodic_mean_rewards = []
        self.index = []
        self.record = record_episodic

        self.num_steps = 0
        self.num_episodes = 0
        self.all_rewards = []
        self.all_dones = []
        self.record_interval = record_interval
        self.interval = step_interval
        self.ep_interval = episode_interval
        self.values = {}
        self.values_idx = {}

        if title is None:
            if hasattr(env, 'spec') and hasattr(env.spec, 'id'):
                title = env.spec.id
            else:
                title = ''
        self.title = title

        if logger is None:
            logger = ch.debug.logger
        self.logger = logger

    def _episodes_length_rewards(self, rewards, dones):
        """
        When dealing with array rewards and dones (as for VecEnv) the length
        and rewards are only computed on the first dimension.
        (i.e. the first sub-process.)
        """
        episode_rewards = []
        episode_lengths = []
        accum = 0.0
        length = 0
        for r, d in zip(rewards, dones):
            if not isinstance(d, bool):
                d = bool(d.flat[0])
                r = float(r.flat[0])
            if not d:
                accum += r
                length += 1
            else:
                episode_rewards.append(accum)
                episode_lengths.append(length)
                accum = 0.0
                length = 0
        if length > 0:
            episode_rewards.append(accum)
            episode_lengths.append(length)
        return episode_rewards, episode_lengths

    def _episodes_stats(self):
        # Find the last episodes
        start = end = count = 0
        for i, d in reversed(list(enumerate(self.all_dones))):
            if not isinstance(d, bool):
                d = d.flat[0]
            if d:
                count += 1
                if end == 0:
                    #moddified index, original of cherry is incorrect
                    end = i
                if count == self.ep_interval + 1:
                    #original of cherry is incorrect
                    start = i + 1
                    break
        # Compute stats
        rewards = self.all_rewards[start:end]
        dones = self.all_dones[start:end]
        rewards, lengths = self._episodes_length_rewards(rewards, dones)
        stats = {
            'episode_rewards': rewards if len(rewards) else [0.0],
            'episode_lengths': lengths if len(lengths) else [0.0],
        }
        return stats

    def stats(self):
        # Compute statistics
        ep_stats = self._episodes_stats()
        steps_stats = self._steps_stats(update_index=True)

        # Overall stats
        if self.record:
            num_logs = self.num_episodes // self.record_interval
        else:
            num_logs = len(self.all_rewards) // self.record_interval

        msg = '-' * 20 + ' ' + self.title + ' Log ' + str(num_logs) + ' ' + '-' * 20 + '\n'
        msg += 'Overall:' + '\n'
        msg += '- Steps: ' + str(self.num_steps) + '\n'
        msg += '- Episodes: ' + str(self.num_episodes) + '\n'

        # Episodes stats
        msg += 'Last ' + str(self.ep_interval) + ' Episodes:' + '\n'
        msg += '- Mean episode length: ' + '%.5f' % mean(ep_stats['episode_lengths'])
        msg += ' +/- ' + '%.5f' % pstdev(ep_stats['episode_lengths']) + '\n'
        msg += '- Mean episode reward: ' + '%.5f' % mean(ep_stats['episode_rewards'])
        msg += ' +/- ' + '%.5f' % pstdev(ep_stats['episode_rewards']) + '\n'
        
        # Steps stats
        msg += 'Last ' + str(self.interval) + ' Steps:' + '\n'
        msg += '- Mean episode length: ' + '%.5f' % mean(steps_stats['episode_lengths'])
        msg += ' +/- ' + '%.5f' % pstdev(steps_stats['episode_lengths']) + '\n'
        msg += '- Mean episode reward: ' + '%.5f' % mean(steps_stats['episode_rewards'])
        msg += ' +/- ' + '%.5f' % pstdev(steps_stats['episode_rewards']) + '\n'
        for key in self.values.keys():
            msg += '- Mean ' + key + ': ' + '%.5f' % mean(steps_stats[key])
            msg += ' +/- ' + '%.5f' % pstdev(steps_stats[key]) + '\n'
        return msg, ep_stats, steps_stats

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def log(self, key, value):
        if key not in self.values:
            self.values[key] = []
            self.values_idx[key] = 0
            setattr(self, key, self.values[key])
        self.values[key].append(value)

    def step(self, *args, **kwargs):
        state, reward, done, info = self.env.step(*args, **kwargs)
        self.all_rewards.append(reward)
        self.all_dones.append(done)
        self.num_steps += 1
        #whether logger generates in episodes or in steps
        if self.record:
            if sum(self.all_dones) > self.num_episodes and self.num_episodes % self.record_interval == 0:
                msg, ep_stats, steps_stats = self.stats()
                self.interval_mean_rewards.append(mean(steps_stats['episode_rewards']))
                self.ci.append(1.96*pstdev(steps_stats['episode_rewards'])/math.sqrt(len(steps_stats['episode_rewards'])))
                self.episodic_mean_rewards.append(mean(ep_stats['episode_rewards']))
                self.epci.append(1.96*pstdev(ep_stats['episode_rewards'])/math.sqrt(len(ep_stats['episode_rewards'])))
                if self.is_vectorized:
                    info[0]['logger_steps_stats'] = steps_stats
                    info[0]['logger_ep_stats'] = ep_stats
                else:
                    info['logger_steps_stats'] = steps_stats
                    info['logger_ep_stats'] = ep_stats
                self.logger.info(msg)
        else:
            if self.interval > 0 and self.num_steps % self.record_interval == 0:
                msg, ep_stats, steps_stats = self.stats()
                self.interval_mean_rewards.append(mean(steps_stats['episode_rewards']))
                self.ci.append(1.96*pstdev(steps_stats['episode_rewards'])/math.sqrt(len(steps_stats['episode_rewards'])))
                self.episodic_mean_rewards.append(mean(ep_stats['episode_rewards']))
                self.epci.append(1.96*pstdev(ep_stats['episode_rewards'])/math.sqrt(len(ep_stats['episode_rewards'])))
                if self.is_vectorized:
                    info[0]['logger_steps_stats'] = steps_stats
                    info[0]['logger_ep_stats'] = ep_stats
                else:
                    info['logger_steps_stats'] = steps_stats
                    info['logger_ep_stats'] = ep_stats
                self.logger.info(msg)

        if isinstance(done, bool):
            if done:
                self.num_episodes += 1
        else:
            self.num_episodes += sum(done)


        return state, reward, done, info