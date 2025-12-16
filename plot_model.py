#! /usr/bin/env python3

# plot viscosity model at 1 depth

import numpy as np
import matplotlib.pyplot as plt
import cartopy as cp
import cartopy.crs as ccrs

irad = 50
clat = np.loadtxt('model/clats.dat')
lons = np.loadtxt('model/elons.dat')
lats = 90.-clat

dvis = np.loadtxt('model/global_dvis_r'+str(irad)+'.dat')
# ====================================== #
# Plot data                              #
# ====================================== #

#cmin = -150.
#cmax = 150.

pmap = dvis
#[ir,ic] = np.where(pmap>=cmax)
#pmap[ir,ic] = cmax
#[ir,ic] = np.where(pmap<cmin)
#pmap[ir,ic] = cmin
    
fig = plt.figure(figsize=(10, 5))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
c = ax.contourf(lons, lats, pmap,
                transform=ccrs.PlateCarree(),
                cmap='RdBu')
fig.colorbar(c)
ax.coastlines()
ax.set_global()
plt.tight_layout()
plt.savefig('vis_map_'+str(irad)+'.pdf')

plt.show()
    



