# -*- coding: utf-8 -*-
"""

Implementation of the ant-system for solving the TSP

http://wjgsp.com/ant-system-python/

Wagner Gonçalves Pinto
Feburary 2021
wjgsp.com

"""

import copy
import inspect
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def euclidean_distance(a,b):
    """ 2D euclidian distance from points a to b """
    if len(a.shape) == 1:
        a, b = np.expand_dims(a,0), np.expand_dims(b,0)
    return np.sqrt(np.sum(np.power(a - b,2.0),axis=1))


def int_euclidean_distance(a,b,dtype=int):
    """ 2D euclidian distance from points a to b """
    return np.array(euclidean_distance(a,b),dtype=dtype)


def att_distance(a,b):
    """ Pseudo-euclidean distance from points a to b """
    if len(a.shape) == 1:
        a, b = np.expand_dims(a,0), np.expand_dims(b,0)
    
    r = np.sqrt(np.sum(np.power(a - b,2.0)/10,axis=1))
    rint = r.astype('int')
    
    distance = rint
    mask = rint < r
    distance[mask] += 1
    
    return distance


def lat_long(angle): 
    """ Converts from angle to geographical latitude and longitude in rad """
    degrees = angle.astype(int)
    minutes = angle - degrees
    return np.pi*(degrees + 5.0*minutes/3.0)/180


def geographical_distance(a,b):
    """ Geographical distance for a idealized earth """
    
    radius = 6378.388 # earth radius in km
    
    if len(a.shape) == 1:
        a, b = np.expand_dims(a,0), np.expand_dims(b,0)
    
    latitude_a, longitude_a = lat_long(a[:,0]), lat_long(a[:,1])
    latitude_b, longitude_b = lat_long(b[:,0]), lat_long(b[:,1])
    
    q1 = np.cos(longitude_a - longitude_b)
    q2 = np.cos(latitude_a - latitude_b)
    q3 = np.cos(latitude_a + latitude_b)
    
    distance = radius*np.acos(
        0.5*(1.0 + q1)*q2 - (1.0-q1)*q3) + 1.0
    
    return distance


class AntSystem():
    """ Ant system for the solution of the Travelling Salesman Problem (TSP)
    
    Implementation as described in Dorigo et al 1996*
    
    *: DORIGO, Marco, MANIEZZO, Vittorio, et COLORNI, Alberto.
       Ant system: optimization by a colony of cooperating agents.
       IEEE Transactions on Systems, Man, and Cybernetics, Part B
       (Cybernetics), 1996, vol. 26, no 1, p. 29-41.
       https://doi.org/10.1109/3477.484436
    
    ARGUMENTS
    ----------
    
        nodes: list or numpy array with the nodes (cities) coordiantes
        distance: matrix with the distances or function to calculate the
                  distance between the nodes
        evaporation:  0.5 . Default is 0.5
        alpha: expoent indicating the importance of trail. Default is 1.0
        beta: expoent for the visibitlity. 5.0 # visibility. Defautl is 5.0
        Q: quantity of laided trail per unit of distance. Defautl is 100.0
        c: initial pheromone level. Default is 0.5 
    
    RETURNS
    ----------
    
        ##none##
    
    """
    def __init__(self,
                 nodes,
                 distance=euclidean_distance,
                 evaporation=0.5,alpha=1.0,beta=5.0,Q=100,
                 c=0.5):
        
        self.float_type = np.float64
        self.nodes =  np.array(nodes,dtype=self.float_type)
        
        self.n = len(self.nodes) # number of nodes
        
        self.integer_type = [np.uint8, np.uint16, np.uint32][
            int(np.log2(self.n) // 8)
            ]
        self.float_type = np.float64
        
        self.evaporation = evaporation
        self.rho = 1 - evaporation
        self.alpha = alpha
        self.beta  =beta
        self.Q = Q
        self.c = c
        
        # array with the distance of nodes
        self.distance = distance
        if inspect.isfunction(self.distance):
            self.dij = np.ones((self.n,self.n),dtype=self.float_type)
            for i in range(self.n): # origin loop
                for j in range(self.n): # destination loop
                    self.dij[i,j] = self.distance(nodes[i,:],nodes[j,:])
        else:
            self.dij = self.distance
            
        # setting diagonal as NaN to remove divide by 0
        self.dij[self.dij == 0.0] = float('NaN')
        visibility = 1/self.dij
        # back to null
        self.dij[np.isnan(self.dij)] = 0.0
        
        self.visibility_beta = visibility**beta
        
        self.paths = define_paths(self.n,integer_type=self.integer_type)
    
    
    def solve(self,_=None,m=None,NC_max=500,):
        """ Performs the ant system optimization of the TSP
        
        ARGUMENTS
        ----------
        
            m: number of ants. Default (None) uses the number of nodes
            NC_max: total number of cycles. Default is 500
        
        RETURNS
        ----------
        
            best_L[-1]: length of the best tour at the end of run
            std_L: standard deviation of the different ants at the last cycle
        
        """
        
        self.NC_max = NC_max
        if m is None:
            m = self.n
        
        # tabu list with visited towns
        tabu = np.zeros((m,self.n),dtype=self.integer_type)
        
        self.trail_intensity = np.zeros(
            (self.n,self.n,self.NC_max),dtype=self.float_type)
        self.trail_intensity[:,:,0] = self.c
        trail_delta = np.zeros(
            (self.n,self.n),dtype=self.float_type)
        # ant, time, origin, destination
        transition_prob = np.ones(
            (m,self.n,self.n),dtype=self.float_type)
        
        list_nodes = [list(range(self.n)) for i in range(m)]
        
        L = np.zeros((m),self.float_type)
        
        self.shortest_tour = np.zeros(
            (self.NC_max,self.n),self.integer_type
            )
        self.best_tour = np.zeros(
            (self.NC_max,self.n),self.integer_type
            )
        self.best_L = np.ones((self.NC_max))*float('Inf')
        
        NC = 0
        
        print('{:^6} {:>12} {:>12} {:>12} {:>12}'.format(
            'cycle','shortest','average','std','best')
            )
        
        # loop of cycles
        while NC < NC_max-1:
            
            allowed_nodes = copy.deepcopy(list_nodes)
            # selecting the starting position of the ants randomly
            tabu[:,0] = self.start_tabu(m)
        
            # loop of ants
            for k in range(m):
                # time in cycle
                for t in range(1,self.n):
                    allowed_nodes[k].remove(tabu[k,t-1])
                    # selected_edge = random.choice(allowed_nodes[k])
                    
                    # probability to go from current edge to allowed destinations
                    probs = transition_prob[k,tabu[k,t-1],allowed_nodes[k]]
                    probs /= np.sum(probs) # normalization
                    
                    selected_edge = self.simple_np_choices(
                        allowed_nodes[k],
                        cum_weights=np.cumsum(probs),
                        )
                    
                    tabu[k,t] = selected_edge
            
                # tour - origin and destination
                tour = np.vstack(
                    ( tabu[k,:], [*tabu[k,1:], tabu[k,0]] )
                    ).T
                # tour length
                L[k] = np.sum(self.dij[tour[:,0],tour[:,1]])
                
                for nodes_index in tour:
                    trail_delta[nodes_index[0],nodes_index[1]] += self.Q/L[k]
                    # if symmetrical!
                    trail_delta[nodes_index[1],nodes_index[0]] += self.Q/L[k]
                
                
                transition_prob[k] = \
                    (self.trail_intensity[:,:,NC]**self.alpha)*self.visibility_beta
                transition_prob[np.isnan(transition_prob)] = 0
                
                
                
            # shortest tour for current cycle
            self.shortest_tour[NC+1,:] = tabu[np.argmin(L),:].copy()
            shortest_L = np.min(L)
            
            if shortest_L < self.best_L[NC]:
                self.best_tour[NC+1,:] = self.shortest_tour[NC+1,:]
                self.best_L[NC+1] = shortest_L
            else:
                self.best_tour[NC+1,:] = self.best_tour[NC]
                self.best_L[NC+1] = self.best_L[NC]
            
            average_L = np.mean(L)
            std_L = np.std(L)
            
            self.trail_intensity[:,:,NC+1] = \
                self.rho*self.trail_intensity[:,:,NC] + trail_delta
            
            # reinstarting tabu list
            tabu *= 0
            trail_delta *= 0
            
            print(f'{NC+1:>6d} '
                  f'{shortest_L:>12.3f} '
                  f'{average_L:>12.3f} '
                  f'{std_L:>12.3f} '
                  f'{self.best_L[NC+1]:>12.3f}')
            sys.stdout.flush()
            
            NC += 1
            
        return self.best_L[-1], std_L
    
    
    def export(self,filename):
        """ Saving npz file with the solution data """
        np.savez(
            filename,
            paths=self.paths,
            trail_intensity=self.trail_intensity,
            best_L=self.best_L,
        )
    
    
    def plot(self,number_cycles=None,
             filename_format='figs/best_tour_NC{:03d}.png',
             title_format='L = {:0.4f}',
             ):
        """ Generating figures with the cycle's and all-time best routes """
        if number_cycles is None:
            # plotting all cycles
            number_cycles = self.trail_intensity.shape[-1]
            
        colormap_name = 'inferno_r'
        colormap = plt.get_cmap(colormap_name)
        colormap = colormap(np.arange(colormap.N))
        # set transparency
        colormap[:,-1] = np.linspace(0, 1, colormap.shape[0])
        colormap = ListedColormap(colormap)
    
        fig, axs = plt.subplots(
            1,2,
            figsize=(12/2.54, 8/2.54),
            constrained_layout=True,
            gridspec_kw=dict(width_ratios=(1.0,0.25)),
            )
        fig.set_constrained_layout_pads(
            hspace=0, wspace=0)
        
        ax_tour, ax_graph = axs[0], axs[1]
        ax_tour.set_anchor('E')
        ax_graph.set_anchor('W')
        
        ax_tour.axis('off')
        path_lines = []
        for path in self.paths:
            x, y = self.nodes[path,0], self.nodes[path,1]
            path_lines.append(
                ax_tour.plot(
                    x,y,
                    '-',
                    color=colormap(1.0),
                    linewidth=3.0,
                    )
                )
        # using a colormap
        # plot normalized pheromone density
        colorbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=None, cmap=colormap),
            ax=ax_tour,
            location='left',
            ticks=[0,0.5,1.0],
            shrink=0.3,
            pad=0.025,
            aspect=20,
            )
        colorbar.ax.set_ylabel('relative trail quantity',fontsize=6)
        colorbar.ax.tick_params(labelsize=6,length=2.0,pad=1.0)
        
        
        ax_graph.plot(
            range(number_cycles),self.best_L[:number_cycles],
            '-r',linewidth=0.75,)
        plot_point, = ax_graph.plot(
            1,self.best_L[1],'+k',markersize=5,linewidth=2.0)
        ax_graph.set_title('tour length',fontsize=6)
        ax_graph.set_xlabel('number of cycles',fontsize=6)
        ax_graph.tick_params(axis='both',length=2.0,labelsize=5,pad=1.0)
        
        # plotting result
        for NC in range(number_cycles-1):
            # updating colors on trails
            for path_index, path in enumerate(self.paths): # origin loop
                index_intensity = [*path,NC]
                max_intensity = np.max(self.trail_intensity[:,:,NC])
                
                path_lines[path_index][0].set_color(
                    colormap(
                        self.trail_intensity[
                            index_intensity[0],index_intensity[1],index_intensity[2]
                            ]/max_intensity
                        ),
                    )
            
            ax_tour, tour_alltime = plotTSP(
                self.nodes,
                IDs=self.best_tour[NC+1],
                ax=ax_tour,
                add_labels=True,
                color='r',
                linewidth=1.5,
                )
            ax_tour, tour_cycle = plotTSP(
                self.nodes,
                IDs=self.shortest_tour[NC+1,:],
                ax=ax_tour,
                color='lightgray',
                style='--',
                linewidth=0.75,
                )
            
            plot_point.set_xdata(NC+1)
            plot_point.set_ydata(self.best_L[NC+1])
            
            
            if NC == 0:
                # fig.legend(
                ax_tour.legend(
                    handles=[tour_cycle[0],tour_alltime[0]],
                    labels=['cycle best','all-time best'],
                    loc='lower left',
                    fancybox=False,
                    markerscale=0,
                    fontsize=6,
                    )
            
            filename = filename_format.format(NC)
            print(filename)
            fig.savefig(filename,dpi=200)
            # remove previous tour
            for tour in (tour_cycle,tour_alltime):
                tour[0].remove()
    
    
    def start_tabu(self,m):
        """ Starting tabu (visited cities) array """
        return np.random.randint(0,high=self.n,size=m,dtype=self.integer_type)
    
    
    def simple_np_choices(self, population, cum_weights, k=1):
        """ Simplified version of the numpy.random.choice function """
        uniform_samples = np.random.random(k)
        return population[int(
                    cum_weights.searchsorted(
                        uniform_samples,
                        side='right'
                        )
                    )
                ]


def plotTSP(nodes,IDs=[],ax=None,
            color='k',style='-',linewidth=1.0,
            add_labels=False):
    """ Function to plot the ants path of a node list.
    
    ARGUMENTS
    ----------
    
        nodes: tuple with the coordinates of the nodes
        IDs: indexes indicating the tour. If empty, plots in the order of nodes
        ax: axis to plot. If None, creates an axis
        color: color of the line
        style: tour plot line style. Optional, default is `-`
        linewidth: tour plot line width. Optional, default is 1.0
        add_labels: Optional, defaults is false
    
    RETURNS
    ---------
    
        ax: axis where the path was plotted
        tour: tour lines plot artist
    
    """
    
    # opening figure
    if ax is None:
        fig, ax = plt.subplots(1)
    marker_size = 8
    
    nodes_map = map_nodes(IDs)
    # converting input into numpy array
    nodes = np.array(nodes, dtype=np.float64)
    
    if add_labels:
        if len(IDs)==0:
            # ploting in the order they are in the list
            pass
        else:
            # adding a text with the index of the cities (if an order
            # is provided) distance for the number plotting
            dx, dy = 0.0, 0.0
            for i in range(len(nodes)):
                ax.text(
                    nodes[i,0]+dx,
                    nodes[i,1]+dy,
                    '{:d}'.format(i),
                    style='normal',
                    fontsize=5,
                    va='center',
                    ha='center',
                    color='w',
                    )
    
    # plotting the nodes
    index_tour = nodes_map[:,0]
    tour = ax.plot(
        nodes[index_tour,0],
        nodes[index_tour,1],
        '-o',
        linestyle=style,
        linewidth=linewidth,
        color=color,
        markerfacecolor='r',
        markersize=marker_size,
        )
    ax.set_aspect('equal')
    
    return ax, tour


def map_nodes(tour):
    """ Full tour considering nodes order """
    return np.vstack(
        ( [*tour,tour[0]], [*tour[1:],*tour[:2]] )
        ).T


def define_paths(n,integer_type=np.int16):
    """ Returns a combination of all posible paths for `n` nodes """
    paths_number = int(n*(n-1)/2)
    paths = np.zeros((paths_number,2),dtype=integer_type)
    
    paths[:,0] = np.hstack(
        [[i]*(n-1-i) for i in range(n-1)],
        )
    paths[:,1] = np.hstack(
        [list(range(i+1,n)) for i in range(n)],
        )
    return paths