#! /usr/bin/env python3

'''
   Extract model to global and regional grid at
   various depth slices
'''

import numpy as np
import matplotlib.pyplot as plt

# info from top of model file:
# data: dvp/vp, dvs/vs, dro/ro, log10(nu3/nu1)
mfile  = '../Data/stp_model_hltm'
odir   = 'model/'
ndepth = 137
nlat   = 361
nlon   = 721

f = open(mfile,'r')
junk = f.readline()
ndepth, nlat, nlon = f.readline().split()
ndepth = int(ndepth)
nlat   = int(nlat)
nlon   = int(nlon)

r      = np.zeros(ndepth)
vismap = np.zeros((ndepth,nlat,nlon))


for i in range(ndepth):
  val = f.readline()
  r[i] = float(val)

f.close()

vals = np.loadtxt(mfile,skiprows=2+ndepth)[:,3]

for i in range(ndepth):
  i0 = i * (nlat*nlon)
  i1 = (i+1) * (nlat*nlon)
  
  data = vals[i0:i1]
  data = np.reshape(data,(nlat,nlon))
  vismap[i,:,:] = data
  
clat = np.linspace(0,180,nlat)
elon = np.linspace(0,360,nlon+1)
elon = elon[0:-1]

np.savetxt(odir+'clats.dat',clat,fmt='%f')
np.savetxt(odir+'elons.dat',elon,fmt='%f')
np.savetxt(odir+'radius.dat',r,fmt='%f')

for i in range(ndepth):
  ofile = odir + 'global_dvis_r'+str(i)+'.dat'
  np.savetxt(ofile,vismap[i,:,:],fmt='%e')
