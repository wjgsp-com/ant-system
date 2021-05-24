Repository with an implementation of the ant system to solve the Travelling Salesman Problem (TSP) as presented in [Dorigo et al. 1996](https://doi.org/10.1109/3477.484436) in Python.

Complete article in:
http://wjgsp.com/ant-system-python/

![Animation of the ant optimization of the Oliver30 case](./antSystem_TSP_Oliver30.gif)

To generate data used to get the routes for the application examples first run `get_routes.py`. [Openrouteservice](https://openrouteservice.org/) is used to obtain the directions and get the driving distance betwen the cities, using their [Python library](https://github.com/GIScience/openrouteservice-py).
Sub-folder `data` contains the list of cities and the csv files with the coordinates for some benchmark problems.

For map production, data is from [Natural Earth](https://www.naturalearthdata.com/) and [GADM database](https://gadm.org/download_country_v3.html); the [cartopy library](https://scitools.org.uk/cartopy/docs/latest/) is used.
