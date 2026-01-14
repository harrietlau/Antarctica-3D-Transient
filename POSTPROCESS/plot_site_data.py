#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

'''
  Import data from the scripts:
  calc_upliftrates1_ANT.py
  calc_upliftrates1_NA.py

  Just easier to extract separately.

'''

AN_rsl_E05 = np.loadtxt('plotdata_RSL_ANT_E05.dat',\
  skiprows=1)
AN_rsl_EA5 = np.loadtxt('plotdata_RSL_ANT_EA5.dat',\
  skiprows=1)
AN_drsl_E05 = np.loadtxt('plotdata_upliftrates_ANT_E05.dat',\
  skiprows=1)
AN_drsl_EA5 = np.loadtxt('plotdata_upliftrates_ANT_EA5.dat',\
  skiprows=1)

NA_rsl_EA5 = np.loadtxt('plotdata_RSL_NA_EA5.dat',\
  skiprows=1)
NA_drsl_EA5 = np.loadtxt('plotdata_upliftrates_NA_EA5.dat',\
  skiprows=1)

NA_rsl_E05 = np.loadtxt('plotdata_RSL_NA_E05.dat',\
  skiprows=1)
NA_drsl_E05 = np.loadtxt('plotdata_upliftrates_NA_E05.dat',\
  skiprows=1)


col1d = '#1f78b4'
col3d = '#33a02c'

plt.figure(1,figsize=(5,8))

# ANTARCTICA PLOTS:
# RSL E05 ANT
plt.subplot(4,2,1)
data = AN_rsl_E05
plt.plot(data[:,0]+2015,data[:,1],'-',color=col1d)
plt.plot(data[:,0]+2015,data[:,2],'--',color=col1d)
plt.plot(data[:,0]+2015,data[:,3],'-',color=col3d)
plt.plot(data[:,0]+2015,data[:,4],'--',color=col3d)

plt.xlim([2000,2300])
plt.xticks([])
plt.ylabel('RSL (m)')
plt.yticks([-20,-10,0])

# DRSL E05 ANT
plt.subplot(4,2,2)
data = AN_drsl_E05
plt.plot(data[:,0]+2015,-data[:,1]*1000.,'-',color=col1d)
plt.plot(data[:,0]+2015,-data[:,2]*1000.,'--',color=col1d)
plt.plot(data[:,0]+2015,-data[:,3]*1000.,'-',color=col3d)
plt.plot(data[:,0]+2015,-data[:,4]*1000.,'--',color=col3d)
plt.legend(['1D','1T','3D','3T'],frameon=False,\
  loc='upper left')

plt.xlim([2000,2300])
plt.xticks([])
plt.ylabel('uplift rate (mm/y)')
plt.yticks([100,200,300])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()


plt.subplot(4,2,3)
# "RSL" assuming 2300 is "present"
data = NA_rsl_E05
#for i in range(len(data[0,:])-1):
#  data[:,i+1] = data[:,i+1]-data[-1,i+1]
plt.plot(data[:,0]+2015,data[:,1],'-',color=col1d)
plt.plot(data[:,0]+2015,data[:,2],'--',color=col1d)
plt.plot(data[:,0]+2015,data[:,3],'-',color=col3d)
plt.plot(data[:,0]+2015,data[:,4],'--',color=col3d)
#plt.ylim([0,140])
#plt.yticks([0.0,0.75,1.5])
plt.xlim([2000,2300])
plt.xticks([2000,2150,2300])
plt.xlabel('year CE')
plt.ylabel('RSL (m)')

# DRSL E05 ANT
plt.subplot(4,2,4)
data = NA_drsl_E05
plt.plot(data[:,0]+2015,-data[:,1]*1000.,'-',color=col1d)
plt.plot(data[:,0]+2015,-data[:,2]*1000.,'--',color=col1d)
plt.plot(data[:,0]+2015,-data[:,3]*1000.,'-',color=col3d)
plt.plot(data[:,0]+2015,-data[:,4]*1000.,'--',color=col3d)
plt.plot([2000,2300],[0,0],'k:',lw=1)
#plt.ylim([0,600])
plt.yticks([-5,0,5,10])
plt.xlim([2000,2300])
plt.xticks([2000,2150,2300])
plt.xlabel('year CE')
plt.ylabel('uplift rate (mm/y)')
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()


# RSL EA5 ANT
plt.subplot(4,2,5)
data = AN_rsl_EA5
plt.plot(data[:,0]+2015,data[:,1],'-',color=col1d)
plt.plot(data[:,0]+2015,data[:,2],'--',color=col1d)
plt.plot(data[:,0]+2015,data[:,3],'-',color=col3d)
plt.plot(data[:,0]+2015,data[:,4],'--',color=col3d)
plt.xlim([2000,2300])
plt.xticks([])
#plt.ylim([0,400])
plt.yticks([-40,-20,0])
plt.ylabel(r'RSL (m)')


plt.subplot(4,2,6)
data = AN_drsl_EA5
plt.plot(data[:,0]+2015,-data[:,1]*1000.,'-',color=col1d)
plt.plot(data[:,0]+2015,-data[:,2]*1000.,'--',color=col1d)
plt.plot(data[:,0]+2015,-data[:,3]*1000.,'-',color=col3d)
plt.plot(data[:,0]+2015,-data[:,4]*1000.,'--',color=col3d)

#plt.ylim([0,600])
plt.xlim([2000,2300])
plt.xticks([])
plt.yticks([150,300,450])
plt.ylabel('uplift rate (mm/y)')
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()



# RSL EA5 ANT
plt.subplot(4,2,7)
data = NA_rsl_EA5
plt.plot(data[:,0]+2015,data[:,1],'-',color=col1d)
plt.plot(data[:,0]+2015,data[:,2],'--',color=col1d)
plt.plot(data[:,0]+2015,data[:,3],'-',color=col3d)
plt.plot(data[:,0]+2015,data[:,4],'--',color=col3d)
plt.xlim([2000,2300])
plt.xticks([2000,2150,2300])
plt.xlabel('year CE')
plt.ylabel(r'RSL (m)')
#plt.ylim([0,400])
#plt.yticks([-2,-1,0])

plt.subplot(4,2,8)
data = NA_drsl_EA5
plt.plot(data[:,0]+2015,-data[:,1]*1000.,'-',color=col1d)
plt.plot(data[:,0]+2015,-data[:,2]*1000.,'--',color=col1d)
plt.plot(data[:,0]+2015,-data[:,3]*1000.,'-',color=col3d)
plt.plot(data[:,0]+2015,-data[:,4]*1000.,'--',color=col3d)
plt.plot([2000,2300],[0,0],'k:',lw=1)

plt.xlabel('year CE')
plt.ylabel('uplift rate (mm/y)')
#plt.ylim([0,600])
plt.xlim([2000,2300])
plt.xticks([2000,2150,2300])
plt.yticks([-20,-10,0,10])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.tight_layout()


plt.savefig('RESULTS/PAPER_FIGURES/time_series.pdf')

# INSETS:

plt.figure(2,figsize=(2,2))
data = AN_rsl_E05
plt.plot(data[:,0]+2015,data[:,1],'-',color=col1d)
plt.plot(data[:,0]+2015,data[:,2],'--',color=col1d)
plt.plot(data[:,0]+2015,data[:,3],'-',color=col3d)
plt.plot(data[:,0]+2015,data[:,4],'--',color=col3d)

plt.xlim([2250,2300])
plt.xticks([2250,2300])
plt.ylabel('RSL (m)')
plt.ylim([-25,-10])
plt.yticks([-25,-10])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.tight_layout()

plt.savefig('RESULTS/PAPER_FIGURES/time_series_inset1_ANE05.pdf')


plt.figure(3,figsize=(2,2))
data = AN_rsl_EA5
plt.plot(data[:,0]+2015,data[:,1],'-',color=col1d)
plt.plot(data[:,0]+2015,data[:,2],'--',color=col1d)
plt.plot(data[:,0]+2015,data[:,3],'-',color=col3d)
plt.plot(data[:,0]+2015,data[:,4],'--',color=col3d)

plt.xlim([2250,2300])
plt.xticks([2250,2300])
plt.ylabel('RSL (m)')
plt.ylim([-60,-25])
plt.yticks([-60,-25])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.tight_layout()

plt.savefig('RESULTS/PAPER_FIGURES/time_series_inset2_ANEA5.pdf')

plt.figure(4,figsize=(2,2))
data = NA_drsl_E05
plt.plot(data[:,0]+2015,-data[:,1]*1000.,'-',color=col1d)
plt.plot(data[:,0]+2015,-data[:,2]*1000.,'--',color=col1d)
plt.plot(data[:,0]+2015,-data[:,3]*1000.,'-',color=col3d)
plt.plot(data[:,0]+2015,-data[:,4]*1000.,'--',color=col3d)

plt.xlim([2250,2300])
plt.xticks([2250,2300])
plt.ylabel('RSL (m)')
plt.ylim([-10,0])
plt.yticks([-10,0])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.tight_layout()

plt.savefig('RESULTS/PAPER_FIGURES/time_series_inset3_NAE05.pdf')

plt.figure(5,figsize=(2,2))
data = NA_drsl_EA5
plt.plot(data[:,0]+2015,-data[:,1]*1000.,'-',color=col1d)
plt.plot(data[:,0]+2015,-data[:,2]*1000.,'--',color=col1d)
plt.plot(data[:,0]+2015,-data[:,3]*1000.,'-',color=col3d)
plt.plot(data[:,0]+2015,-data[:,4]*1000.,'--',color=col3d)

plt.xlim([2250,2300])
plt.xticks([2250,2300])

plt.ylabel('RSL (m)')
plt.ylim([-30,-10])
plt.yticks([-30,-10])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.tight_layout()

plt.savefig('RESULTS/PAPER_FIGURES/time_series_inset4_NAEA5.pdf')

plt.figure(6,figsize=(1.5,2))
data = NA_rsl_E05
#for i in range(len(data[0,:])-1):
#  data[:,i+1] = data[:,i+1]-data[-1,i+1]

plt.plot([1,2],[data[3,1],data[3,1]],'-',color=col1d,lw=3)
plt.plot([1,2],[data[3,2],data[3,2]],'--',color=col1d,lw=3)
plt.plot([1,2],[data[3,3],data[3,3]],'-',color=col3d,lw=3)
plt.plot([1,2],[data[3,4],data[3,4]],'--',color=col3d,lw=3)

plt.xlim([0.5,2.5])
plt.xticks([1.5],['2015'])
plt.ylabel(r'$\Delta$')
plt.yticks([-0.25,-0.18])
plt.ylim([-0.25,-0.18])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.tight_layout()

plt.savefig('RESULTS/PAPER_FIGURES/time_series_inset5_NAE05.pdf')

plt.figure(7,figsize=(1.5,2))
data = NA_rsl_EA5
#for i in range(len(data[0,:])-1):
#  data[:,i+1] = data[:,i+1]-data[-1,i+1]

plt.plot([1,2],[data[3,1],data[3,1]],'-',color=col1d,lw=3)
plt.plot([1,2],[data[3,2],data[3,2]],'--',color=col1d,lw=3)
plt.plot([1,2],[data[3,3],data[3,3]],'-',color=col3d,lw=3)
plt.plot([1,2],[data[3,4],data[3,4]],'--',color=col3d,lw=3)

plt.xlim([0.5,2.5])
plt.xticks([1.5],['2015'])
plt.ylabel(r'$\Delta$')
plt.yticks([-0.25,-0.18])
plt.ylim([-0.25,-0.18])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.tight_layout()

plt.savefig('RESULTS/PAPER_FIGURES/time_series_inset6_NAEA5.pdf')





plt.show()

