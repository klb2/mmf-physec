# Based on the waterfilling implementation by Kenneth Shum, 2010.
# https://www.mathworks.com/matlabcentral/fileexchange/28022-waterfilling-algorithm
#
# Reference:
#   T. M. Cover and J. A. Thomas, "Elements of Information Theory", John Wiley
#   & Sons, 2003.

import numpy as np

def water_filling(noise_power, power):
    n = len(noise_power)
    _sort_power = np.sort(noise_power)
    delta = np.hstack(([0], np.cumsum((_sort_power[1:]-_sort_power[:-1])*np.arange(1, n))))
    l = np.count_nonzero(power >= delta)
    level = _sort_power[l-1] + (power-delta[l-1])/l
    power_vec = (np.abs(level - noise_power) + level - noise_power)/2
    return power_vec

if __name__ == "__main__":
    n = 5
    a = np.array([1.2, 2.2, 3.5, 1, 0.5])
    power = 10
    print(water_filling(a, 10))
