#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

# =============================================== #
# inputs:
# file names:
fdir  = 'RESULTS/EM'
fruns = ['E05','EA5','NCR','F32']
fmods = ['1D', '1T', '3D', '3T' ]
flast = ['57', '57', '42', '42' ]
fgmsl = ['exp05.dat','expA5.dat',\
         'NCR.dat','F32.dat']

# site choices
osite = ['New Orleans',\
         'DC',\
         'London',\
         'Lagos',\
         'Tokyo']

slat    = np.array([29.9509,\
                    38.9072,\
                    51.5072,\
                    6.6137,\
                    35.6764])
                    
slon    = np.array([-90.0758,\
                    -77.0369,\
                    0.1276,\
                    3.3553,\
                    139.6500])
nsite   = len(osite)

# load lat/lon
clat  = np.loadtxt('RESULTS/EM1D_E05/clat.dat')
lat   = 90.-clat
lon   = np.loadtxt('RESULTS/EM1D_E05/elon.dat')

nrun  = len(fruns)
nmod  = len(fmods)
rslE05 = []
rslEA5 = []
rslNCR = []
rslF32 = []

gmsl  = np.zeros(nrun)

for i in range(nmod):
  # load maps
  fname = fdir + fmods[i] + '_' + fruns[0]+'/' \
    'RSL_' + flast[0]
  vals = np.loadtxt(fname)
  rslE05.append(vals)
  fname = fdir + fmods[i] + '_' + fruns[1]+'/' \
    'RSL_' + flast[1]
  vals = np.loadtxt(fname)
  rslEA5.append(vals)
  fname = fdir + fmods[i] + '_' + fruns[2]+'/' \
    'RSL_' + flast[2]
  vals = np.loadtxt(fname)
  rslNCR.append(vals)
  fname = fdir + fmods[i] + '_' + fruns[3]+'/' \
    'RSL_' + flast[3]
  vals = np.loadtxt(fname)
  rslF32.append(vals)
  

for i in range(nrun):  # load GMSL
  fname = 'GMSL/GMSL_'+fgmsl[i]
  vals = np.loadtxt(fname)
  gmsl[i] = vals[-1,-1]

# now interpolate

rslsE05 = np.zeros((nmod,nsite))
rslsEA5 = np.zeros((nmod,nsite))
rslsNCR = np.zeros((nmod,nsite))
rslsF32 = np.zeros((nmod,nsite))

for i in range(nmod):
  f1 = si.interp2d(lon,lat,rslE05[i])
  f2 = si.interp2d(lon,lat,rslEA5[i])
  f3 = si.interp2d(lon,lat,rslNCR[i])
  f4 = si.interp2d(lon,lat,rslF32[i])
  for j in range(nsite):
    rslsE05[i,j] = f1(slon[j],slat[j])
    rslsEA5[i,j] = f2(slon[j],slat[j])
    rslsNCR[i,j] = f3(slon[j],slat[j])
    rslsF32[i,j] = f4(slon[j],slat[j])


xmid = np.arange(nsite)
x0   = xmid[0]-0.5
x1   = xmid[-1]+0.5
dx1  = -0.15
dx2  = -0.05
dx3  = +0.05
dx4  = +0.15

col1 = '#1f78b4'
col2 = '#33a02c'

# plot

plt.figure(1,figsize=(10,5))

plt.subplot(2,2,1)
irun = 2 # NCR
plt.bar(xmid+dx1,rslsNCR[0,:]-gmsl[irun],width=0.06,color=col1)
plt.bar(xmid+dx2,rslsNCR[1,:]-gmsl[irun],width=0.06,fill=False,hatch='//',ec=col1)
plt.bar(xmid+dx3,rslsNCR[2,:]-gmsl[irun],width=0.06,color=col2)
plt.bar(xmid+dx4,rslsNCR[3,:]-gmsl[irun],width=0.06,fill=False,hatch='//',ec=col2)
plt.plot([x0,x1],[0,0],'k--',lw=0.5)
plt.xlim([x0,x1])
plt.xticks([])
plt.yticks([0.0,0.05,0.1,0.15])
plt.ylabel('$\Delta$GMSL (m)')

plt.subplot(2,2,2)
irun = 0 #E05
plt.bar(xmid+dx1,rslsE05[0,:]-gmsl[irun],width=0.06,color=col1)
plt.bar(xmid+dx2,rslsE05[1,:]-gmsl[irun],width=0.06,fill=False,hatch='//',ec=col1)
plt.bar(xmid+dx3,rslsE05[2,:]-gmsl[irun],width=0.06,color=col2)
plt.bar(xmid+dx4,rslsE05[3,:]-gmsl[irun],width=0.06,fill=False,hatch='//',ec=col2)
plt.xticks([])
plt.xlim([x0,x1])

plt.yticks([0.2,0.4,0.6,0.8])
plt.ylabel('$\Delta$GMSL (m)')
ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.subplot(2,2,3)
irun = 3 # F32
plt.bar(xmid+dx1,rslsF32[0,:]-gmsl[irun],width=0.06,color=col1)
plt.bar(xmid+dx2,rslsF32[1,:]-gmsl[irun],width=0.06,fill=False,hatch='//',ec=col1)
plt.bar(xmid+dx3,rslsF32[2,:]-gmsl[irun],width=0.06,color=col2)
plt.bar(xmid+dx4,rslsF32[3,:]-gmsl[irun],width=0.06,fill=False,hatch='//',ec=col2)
plt.xlim([x0,x1])

plt.yticks([0.05,0.10,0.15,0.2])
plt.xticks(xmid,osite)

plt.ylabel('$\Delta$GMSL (m)')


plt.subplot(2,2,4)
irun = 1 #EA5
plt.bar(xmid+dx1,rslsEA5[0,:]-gmsl[irun],width=0.06,color=col1)
plt.bar(xmid+dx2,rslsEA5[1,:]-gmsl[irun],width=0.06,fill=False,hatch='//',ec=col1)
plt.bar(xmid+dx3,rslsEA5[2,:]-gmsl[irun],width=0.06,color=col2)
plt.bar(xmid+dx4,rslsEA5[3,:]-gmsl[irun],width=0.06,fill=False,hatch='//',ec=col2)
plt.yticks([0.3,0.6,0.9,1.2])
plt.xlim([x0,x1])
plt.xticks(xmid,osite)
plt.ylabel('$\Delta$GMSL (m)')

ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")


plt.tight_layout()


plt.figure(2,figsize=(10,5))

plt.subplot(2,2,1)
irun = 2 # NCR
plt.plot([0,0],[0,0],'k-')
plt.legend(['GMSL = 0.11 m'],frameon=False)
plt.bar(xmid+dx1,100*(rslsNCR[0,:]-gmsl[irun])/gmsl[irun],width=0.06,color=col1)
plt.bar(xmid+dx2,100*(rslsNCR[1,:]-gmsl[irun])/gmsl[irun],width=0.06,fill=False,hatch='//',ec=col1)
plt.bar(xmid+dx3,100*(rslsNCR[2,:]-gmsl[irun])/gmsl[irun],width=0.06,color=col2)
plt.bar(xmid+dx4,100*(rslsNCR[3,:]-gmsl[irun])/gmsl[irun],width=0.06,fill=False,hatch='//',ec=col2)
plt.plot([x0,x1],[0,0],'k--',lw=0.5)
plt.xlim([x0,x1])
plt.xticks([])
plt.yticks([-15,0,15,30,45])
plt.ylabel('$\Delta$GMSL (%)')

plt.subplot(2,2,2)
irun = 0 #E05
plt.plot([0,0],[0,0],'k-')
plt.legend(['GMSL = 1.53 m'],frameon=False)
plt.bar(xmid+dx1,100*(rslsE05[0,:]-gmsl[irun])/gmsl[irun],width=0.06,color=col1)
plt.bar(xmid+dx2,100*(rslsE05[1,:]-gmsl[irun])/gmsl[irun],width=0.06,fill=False,hatch='//',ec=col1)
plt.bar(xmid+dx3,100*(rslsE05[2,:]-gmsl[irun])/gmsl[irun],width=0.06,color=col2)
plt.bar(xmid+dx4,100*(rslsE05[3,:]-gmsl[irun])/gmsl[irun],width=0.06,fill=False,hatch='//',ec=col2)
plt.xticks([])
plt.xlim([x0,x1])

plt.yticks([0,5,10,15,20])
plt.ylabel('$\Delta$GMSL (%)')
ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.subplot(2,2,3)
irun = 3 # F32
plt.plot([0,0],[0,0],'k-')
plt.legend(['GMSL = 0.3 m'],frameon=False)
plt.bar(xmid+dx1,100*(rslsF32[0,:]-gmsl[irun])/gmsl[irun],width=0.06,color=col1)
plt.bar(xmid+dx2,100*(rslsF32[1,:]-gmsl[irun])/gmsl[irun],width=0.06,fill=False,hatch='//',ec=col1)
plt.bar(xmid+dx3,100*(rslsF32[2,:]-gmsl[irun])/gmsl[irun],width=0.06,color=col2)
plt.bar(xmid+dx4,100*(rslsF32[3,:]-gmsl[irun])/gmsl[irun],width=0.06,fill=False,hatch='//',ec=col2)
plt.xlim([x0,x1])

plt.yticks([0,5,10,15,20])
plt.xticks(xmid,osite)

plt.ylabel('$\Delta$GMSL (%)')


plt.subplot(2,2,4)
irun = 1 #EA5
plt.plot([0,0],[0,0],'k-')
plt.legend(['GMSL = 3.47 m'],frameon=False)
plt.bar(xmid+dx1,100*(rslsEA5[0,:]-gmsl[irun])/gmsl[irun],width=0.06,color=col1)
plt.bar(xmid+dx2,100*(rslsEA5[1,:]-gmsl[irun])/gmsl[irun],width=0.06,fill=False,hatch='//',ec=col1)
plt.bar(xmid+dx3,100*(rslsEA5[2,:]-gmsl[irun])/gmsl[irun],width=0.06,color=col2)
plt.bar(xmid+dx4,100*(rslsEA5[3,:]-gmsl[irun])/gmsl[irun],width=0.06,fill=False,hatch='//',ec=col2)
plt.yticks([0,5,10,15,20])
plt.xlim([x0,x1])
plt.xticks(xmid,osite)
plt.ylabel('$\Delta$GMSL (%)')

ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")


plt.tight_layout()

plt.savefig('RESULTS/PAPER_FIGURES/rsl_bchart.pdf')

plt.show()
