"""
Copyright (C) 2023 Karl-Ludwig Besser

This program is used in the article:
"Securing Data in Multimode Fibers by Exploiting Mode-Dependent Light
Propagation Effects" (S. Rothe, K.-L. Besser, D. Krause, R. Kuschmierz, N.
Koukourakis, E. Jorswieck, J. Czarske. Research, vol. 6: 0065, Jan. 2023.
DOI:10.34133/research.0065).


License:
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
See the GNU General Public License for more details.

Author: Karl-Ludwig Besser, Technische Universität Braunschweig
"""

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
