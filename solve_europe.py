# -*- coding: utf-8 -*-
"""

Solving and plotting the TSP optimization problem using ant system
for a route connecting the european capitals. Reads npz file
generated by `get_routes.py`

http://wjgsp.com/ant-system-python/

Wagner Gonçalves Pinto
Feburary 2021
wjgsp.com

"""

import numpy as np

import cartopy.feature
import cartopy.crs as ccrs
from cartopy.io import shapereader
import matplotlib.pyplot as plt

import ant_system

if __name__ == '__main__':
    
    # parameters
    parameters = dict(
        evaporation = 0.5, # (ref. 0.5)
        alpha = 1.0, # importance of trail (ref. 1)
        beta = 5.0, # visibility (ref. 5)
        Q = 100, # quantity of laided trail (ref. 100)
        c = 0.5, # initial pheromone level
    )
    
    
    filename = 'data/routes_europe_capitals_n48.npz'
    central_lon, central_lat = 20, 47.5
    extent = [-10, 50, 34, 61]
    
    
    data = np.load(filename,allow_pickle=True)
    asystem = ant_system.AntSystem(
                    nodes=data['coordinates'],
                    distance=data['dij'],
                    **parameters
                    )
    print('\nOptimizing route...')
    asystem.solve(m=2*len(data['coordinates']), NC_max=1000)
    
    initial_tour = asystem.best_tour[1,:]
    final_tour = asystem.best_tour[-1,:]
    
    
    print('\nPlotting map...')
    fig = plt.figure(figsize=(12,7),constrained_layout=True)
    
    ax = plt.axes(
        projection=cartopy.crs.Robinson(central_longitude=central_lon))
    
    ax.set_extent(extent)
    ax.coastlines(resolution='50m')
    ax.add_feature(cartopy.feature.LAND, color='papayawhip')
    ax.add_feature(cartopy.feature.OCEAN, color='lightcyan')
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='gray')
    
    route_handles = []
    for tour, line in zip(
            (initial_tour, final_tour),(':r','-b')):
        
        for origin, destination in ant_system.map_nodes(tour):
            path = [origin,destination]
            path.sort()
            try:
                route_index = data['paths'].tolist().index(path)
            except(ValueError):
                continue
            
            route = data['routes'][route_index]
            route_plot = ax.plot(
                route[:,0],
                route[:,1],
                line,
                linewidth=1.5,
                transform=ccrs.Geodetic(),
                )
        route_handles.append(route_plot[0])
        
    ax.legend(
        handles=route_handles,
        labels=(f'initial route ({asystem.best_L[1]:,.0f} km)',
                f'best route ({asystem.best_L[-1]:,.0f} km)'
                )
        )
    
    # coloring the countries
    shapefilename = shapereader.natural_earth(
        resolution='50m', category='cultural', name='admin_0_countries')
    reader = shapereader.Reader(shapefilename)
    countries_info = list(reader.records())
    countries_names = np.array(
        [country.attributes['NAME']
         for country in countries_info],dtype=object
        )
    
    locations_to_plot = data['locations'][:,-1]
    locations_indexes = np.zeros(locations_to_plot.shape,dtype=np.int16)
    colormap = plt.cm.get_cmap('inferno',len(locations_to_plot))
    # getting the indexes of the countries of interest
    for index_loc, loc in enumerate(locations_to_plot):
        
        if loc == 'Bosnia and Herzegovina': loc = 'Bosnia and Herz.'
        if loc == 'North Macedonia': loc = 'Macedonia'
        if loc == 'Vatican City (Holy See)': loc = 'Vatican'
        
        index = int(np.where(countries_names == loc)[0])
        locations_indexes[index_loc] = index
        ax.add_geometries([countries_info[index].geometry],
                          crs=ccrs.PlateCarree(),
                          facecolor=colormap(index_loc),
                          edgecolor='gray',
                          linewidth=2.0,
                          alpha=0.4)
    
    ax.gridlines(linewidth=0.5,color='lightgray')
    
    cities_label_left = np.array(
        ('Lisbon','Rome','Zagreb','Vaduz','Budapest',
         'Moscow','Tallinn','Bratislava','Tbilisi',
         'Sofia','Skopje'),
        dtype=object
        )
    for index_coord, coord in enumerate(data['coordinates']):
        ax.plot(
            coord[0],coord[1],
            'ok',
            markersize=5,
            transform=ccrs.Geodetic(),
        )
        
        city_name = data['locations'][index_coord][0]
        # 'manually' placing some labels
        horizontal_alignement = 'right'
        offset = -0.25
        if np.any(city_name == cities_label_left):
            horizontal_alignement = 'left'
            offset = +0.25
            
        ax.text(
            coord[0]+offset,coord[1],
            city_name,
            fontsize=8.5,
            va='center',ha=horizontal_alignement,
            bbox={'facecolor': 'gray', 'alpha': 0.8, 'pad': 1},
            transform=ccrs.Geodetic(),
            )
    
    print('Done')
    fig.savefig('figs/route_europe.png',dpi=290)