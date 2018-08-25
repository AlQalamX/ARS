# ARS AI 08-25-2018

import os
import numpy as np
import gym
from gym import wrappers
# setting the hyperparameters (usually a param of a fixed value)

class Hp():
    def __init__(self):
        self.numb_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.numb_directions = 16
        self.numb_best_directions = 16
        assert self.numb_best_directions <= self.numb_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = ''

# Normalize the states 
        
class Normalizer():
    
    def __init__(self, numb_inputs):
        self.n = np.zeros(numb_inputs)
        self.mean = np.zeros(numb_inputs)
        self.mean_diff = np.zeros(numb_inputs)
        self.var = np.zeros(numb_inputs)
        
# Method that updates and computes the mean and variance everytime a new state is observed
# mean = sum of values / numb of values      
    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        #compute the variance(numerator)
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        
    def normailize(self, inputs):
        obs_mean = self.mean
        obs_stand_deviat = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_stand_deviat

#build the ai

class AI():

    def __init__(self, input_size, output_size):
        # weight matrix .........left multiplication
        self.theta = np.zeros((output_size, input_size))

    # apply pertubations on weight matrix

    '''
    returns output when fed an input
    returns output when fed an input and + direction perturbation is applied
    returns output when fed an input and - direction perturbation is applied
    '''
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise * delta).dot(input)
        else:
            return (self.theta - hp.noise * delta).dot(input)
        
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.numb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delt in rollouts:
            step += (r_pos - r_neg) * delt
        self.theta += hp.learning_rate / (hp.numb_best_directions * sigma_r) * step

# explore the policy on one specific direction and over one episode
# allows for reward to be compared after a complete episode --optimal decision making/exploit 
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    numb_plays = 0.
    sum_rewards = 0
    while not done and numb_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normailize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        numb_plays += 1
    return sum_rewards

#Training th AI

def train(env, policy, normalizer, hp):
    
    for step in range(hp.numb_steps):

        #Initializing the perturbations deltas & the pos/neg rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.numb_directions
        negative_rewards = [0] * hp.numb_directions

        # Getting positive rewards in the positive directions

        for k in range(hp.numb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])

        # Getting negative rewards in the opposite directions

        for k in range(hp.numb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])

        # Gather all pos/neg rewards into one list to compute standard deviation of rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(),key = lambda x:scores[x])[0:hp.numb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating the policy... Section 3 Algorithm v2 step 7
        policy.update(rollouts, sigma_r)
            
        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print('Step: ', step, ' Reward: ', reward_evaluation)

# Running the main code

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'bra')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env,monitor_dir, force = True)
numb_inputs = env.observation_space.shape[0]
numb_outputs = env.action_space.shape[0]
policy = AI(numb_inputs, numb_outputs)
normalizer = Normalizer(numb_inputs)
train(env, policy, normalizer, hp)


 