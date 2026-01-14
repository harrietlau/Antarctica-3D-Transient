#! /usr/bin/env python3

'''
   Plot GMSL from each ice model
   
   Don't think the LIG to present has
   the 1900-2014 model. check with KL.
   
'''

import numpy as np
import matplotlib.pyplot as plt

[tlig,sllig] = np.loadtxt('EUi6g.dat',unpack=True)
[tshro,sshro] = np.loadtxt('EUt_ANT.dat',unpack=True) # antarctic component for the full history including 1900-2015 (shroeder)
[t1,sl1]     = np.loadtxt('GMSL_exp05.dat',unpack=True)
[t2,sl2]     = np.loadtxt('GMSL_expA5.dat',unpack=True)
[t3,sl3]     = np.loadtxt('GMSL_NCR.dat',unpack=True)
[t4,sl4]     = np.loadtxt('GMSL_F32.dat',unpack=True)

cols = ['#e66101','#fdb863','#5e3c99','#b2abd2']


tlig  = tlig * 1e3 # make everything years
tshro = tshro * 1e3

# do some stitching:
tshro = tshro + 2014
idx   = np.where((tshro>=1900))[0]
tshro = tshro[idx]
sshro = sshro[idx]

# assume everything rising from LIG to present:
tlig  = tlig[1:len(tlig)]
sllig = sllig[1:len(sllig)]
sllig = sllig - sshro[-1]

# concatenate:
tpast  = np.concatenate((tlig,tshro))
slpast = np.concatenate((sllig,sshro))

t1900  = tshro
sl1900 = sshro

# stitch modern time
sl1pres = np.concatenate((sl1900,sl1))
sl2pres = np.concatenate((sl1900,sl2))
sl3pres = np.concatenate((sl1900,sl3))
sl4pres = np.concatenate((sl1900,sl4))

# get time steps:
#tlig1 = (tlig[0:-1]+tlig[1::])*0.5
#dt    = np.diff(tlig)
# from konstantin:
[t_dt,dt] = np.loadtxt('dtt.dat',unpack=True)

plt.figure(1,figsize=(7,3))
plt.subplot(1,2,1)
plt.plot(tpast/1000.,slpast,'k-',lw=2)
plt.xlabel(r'$t$ (ky)')
plt.ylabel('GMSL (m)')
plt.fill_betweenx(np.array([-130,0]), 0, 1.9, color='gray', alpha=0.5)

plt.xlim([-122,2])
plt.xticks([-120,-80,-40,0])
plt.ylim([-130,0])
plt.yticks([-100,-50,0])
ax1 = plt.gca()

ax2 = ax1.twinx()
#ax2.plot(tlig1/1000.,dt,'k--',lw=1)
ax2.plot(t_dt,dt,'k--',lw=1)
ax2.set_ylabel(r'$\Delta t$ (ky)')

plt.subplot(1,2,2)
plt.fill_betweenx(np.array([-1,4]), 1900, 2014, color='gray', alpha=0.5)
plt.plot(t1,sl1,'--',lw=2,color=cols[0])
plt.plot(t2,sl2,'-',lw=2,color=cols[1])
plt.plot(t3,sl3,'--',lw=2,color=cols[2])
plt.plot(t4,sl4,'-',lw=2,color=cols[3])

plt.plot(t1900,sl1900,'k-',lw=2)
plt.xlabel(r'$t$ (CE)')
plt.ylabel('GMSL (m)')
plt.xlim([1900,2300])
plt.xticks([1900,2000,2100,2200,2300])
plt.ylim([-1,4])
plt.yticks([0,1,2,3,4])


ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.tight_layout()

plt.savefig('gmsl.pdf')

plt.figure(2,figsize=(1.5,1.5))
plt.plot(t3,sl3,'--',lw=2,color=cols[2])
plt.plot(t4,sl4,'-',lw=2,color=cols[3])
plt.xticks([2015,2100])
plt.ylim([0,0.4])
plt.yticks([0.2,0.4])
ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.tight_layout()

plt.savefig('gmsl_inset.pdf',transparent=True)


plt.show()
