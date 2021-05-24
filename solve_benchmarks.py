# -*- coding: utf-8 -*-
"""

Solution of the TSP with ant system for benchmarks.
Many cases are available on TSPLIB:
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

http://wjgsp.com/ant-system-python/

Wagner Gonçalves Pinto
Feburary 2021
wjgsp.com

"""

import numpy as np

import ant_system

if __name__ == '__main__':
    
    parameters = dict(
        evaporation = 0.5, # (ref. 0.5)
        alpha = 1.0, # importance of trail (ref. 1)
        beta = 5.0, # visibility (ref. 5)
        Q = 100, # quantity of laided trail (ref. 100)
        c = 0.5, # initial pheromone level
    )
    
    case_name = 'Berlin52'
    nodes = np.genfromtxt(f'data/{case_name}.csv', delimiter=',')
    fun_dist = ant_system.int_euclidean_distance
    ref_L = 7542
    title_format='{:>5d}, L = {:d}'
    
    case_name = 'Oliver30'
    nodes = np.genfromtxt(f'data/{case_name}.csv', delimiter=',')
    fun_dist = ant_system.euclidean_distance
    ref_L = 423.741
    title_format=r'cycle {:>5d}   L = {:0.2f}'
    
    n = len(nodes) # number of nodes
    m = int(n*2) # total number of ants
    
    n_runs = 1
    NC_max = 500
    
    asystem = ant_system.AntSystem(
        nodes=nodes,
        distance=fun_dist,
        **parameters
        )
    
    print(f'Run for {case_name:}...')
    asystem.solve(m=m,NC_max=NC_max)
    print('Generating images...')
    asystem.plot(
        filename_format='figs/best_' + case_name + 'tour_NC{:03d}.png',
        number_cycles=300,
        title_format=title_format,
        )
    
    