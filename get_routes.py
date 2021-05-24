# -*- coding: utf-8 -*-
"""

Getting the routes between the pairs of destinations cities and building
the distance matrix `dij` considering a symmetrical route (going from 
A to B equals going from B to A). Saves a npz file.

In order to use it you need to get your own Openrouteservices API
(https://openrouteservice.org/)

http://wjgsp.com/ant-system-python/

Wagner Gonçalves Pinto
Feburary 2021
wjgsp.com

"""

import time
import csv

import numpy as np
import openrouteservice

import ant_system


if __name__ == '__main__':
    
    with open('openrouteservice_api_key', 'r') as f:
        my_API_key = f.readline() # specify your personal API key
    
    client = openrouteservice.Client(key=my_API_key)
    
    # case, decrypt_loc = 'europe_capitals', lambda loc: dict(
    #     text=loc[0],country=loc[1])
    
    # case, decrypt_loc = 'brazil_capitals', lambda loc: dict(
    #      text=f'{loc[0]},{loc[1]}',country=loc[-1],layers=['locality,region'])
    
    case, decrypt_loc = 'france_capitals', lambda loc: dict(
         text=f'{loc[0]},{loc[1]}',country=loc[-1],layers=['locality,region'])
    
    
    with open(f'data/{case}.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        locations = [row for row in reader]
    
    n = len(locations)
    
    coordinates = []
    for loc in locations:
        # geocoding - pelias
        result = client.pelias_search(
            **decrypt_loc(loc),
            )
        coordinates.append(
            result['features'][0]['geometry']['coordinates']
            )
    coordinates = np.array(coordinates)
    
    paths = ant_system.define_paths(n,np.int16)
    paths = paths.tolist()
    
    # calculating the routes for each pair origin - destination
    # one can also use client.distance_matrix(coordinates.tolist()) to get
    # the matrix of distances directly, but the routes between the
    # destinations will not be available
    routes, distances, durations = [], [], []
    impossible_paths = []
    for index_path, path in enumerate(paths):
        print(f'{path[0]}-{path[1]}'
              f' {locations[path[0]][0]} > {locations[path[1]][0]}')
        try:
            route = client.directions(
                coordinates=coordinates[path,:].tolist(),
                profile='driving-car',
                radiuses=[-1]*len(path),
                units='km',
                )
            
            geometry = route['routes'][0]['geometry']
            decoded = openrouteservice.convert.decode_polyline(geometry)
            
            durations.append(route['routes'][0]['summary']['duration'])
            distances.append(route['routes'][0]['summary']['distance'])
            routes.append(np.array(decoded['coordinates']))
        
        # ignoring infeasible trails - may also stop if number of
        # requests limit is reached
        except(openrouteservice.exceptions.ApiError):
            print(' >> Route could not be found')
            impossible_paths.append(path)
            
        # limited to 40 requests per minute
        time.sleep(60/40)
    
    for path in impossible_paths:
        paths.remove(path)
    
    # exporting the distance matrix - symmetrical
    dij = np.zeros((n,n),dtype=np.float64)
    for index_path, path in enumerate(paths):
        i, j = path
        dij[i,j] = distances[index_path]
        dij[j,i] = dij[i,j]
    
    # converting locations to numpy array to save data in npz file format
    locations_arr = np.array(locations,dtype=object)
    
    # exporting the routes
    np.savez(
        f'data/routes_{case}_n{n}.npz',
        locations=locations_arr,
        coordinates=np.array(coordinates),
        paths=np.array(paths),
        routes=routes,
        distances=distances,
        durations=durations,
        dij=dij,
        )