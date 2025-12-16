# ============================================= #
# functions to plot Seakon output               #
# ============================================= #

import numpy as np
import scipy.interpolate as si

def gl(n):
    # gets the GL nodes
    
    # input:  number of GL (base 2)
    # output: colatitudes from S to N in radians
    
    x,w = np.polynomial.legendre.leggauss(n)
    cl  = np.arccos(x)

    return cl


def get_map(vals,n):
    # convert Seakon column output to
    # grid.
    # outputs S to N and 0 to 360.

    vmap = np.reshape(vals,(n,2*n))
    vmap = np.flipud(vmap)
    clat = gl(n)
    elon = np.linspace(0,2.*np.pi,2*n)

    return clat,elon,vmap

def xyz2latlon(x,y,z):
    val  = z/np.sqrt(x*x + y*y + z*z)
    clat = np.arccos(val)
    val  = x/np.sqrt(x*x + y*y)
    elon = np.sign(y)*np.arccos(val)

    lat  = 90.-clat*180./np.pi
    lon  = elon*180./np.pi

    outmat = np.zeros((len(x),2))
    outmat[:,0] = lon
    outmat[:,1] = lat
        
    return outmat
    
    
def get_ant_map(vals,ll0,lat1,lon1):
    # interpolate Seakon regional output
    # to regular grid
    # ll0 : 2 columns of lons and lats of
    #       original file
    # lat1: new latitude points
    # lon1: new longitude points    

    nlat = len(lat1)
    nlon = len(lon1)
    ll1  = np.zeros((nlat*nlon,2))
    idx  = 0
    
    for i in range(nlat):
        for j in range(nlon):
            ll1[idx,0] = lon1[j]
            ll1[idx,1] = lat1[i]
            idx = idx + 1

    vals1 = si.griddata(ll0,vals,ll1)

    mmap  = np.reshape(vals1,(nlat,nlon))

    return mmap
