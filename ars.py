# ARS AI 08-25-2018

import os
import numpy as np

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

    
            
 