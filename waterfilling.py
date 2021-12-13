import numpy as np
from scipy import optimize, stats

def binary_entropy(prob):
    return -(prob*np.log(prob) + (1-prob)*np.log(1-prob))

def cost_func(x, sigma):
    return np.sum(binary_entropy(stats.norm.sf(sigma*x)))

def water_filling_bsc(sigma, power):
    num_channels = len(sigma)
    constraint = optimize.LinearConstraint(np.ones(num_channels), power, power)
    x0 = power/num_channels * np.ones(num_channels)
    sol = optimize.minimize(cost_func, x0, args=(sigma,),
                            bounds=[[0, power]]*num_channels,
                            constraints=constraint)
    return sol.x
