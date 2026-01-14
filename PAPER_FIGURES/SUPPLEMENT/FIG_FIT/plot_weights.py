#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mfunc as M


[w1,t1] = np.loadtxt('gmx_seakon10.dat',unpack=True,\
  skiprows=1)
[w2,t2] = np.loadtxt('gmx_seakon20.dat',unpack=True,\
  skiprows=1)
[w3,t3] = np.loadtxt('gmx_seakon50.dat',unpack=True,\
  skiprows=1)

nf  = 500
w   = np.logspace(-20,20,nf)
s   = w
t   = 1./w

tM  = 1.
mu  = 1.
Ju  = 1./mu
nu  = 1.

# Maxwell - Fourier
Mmx_w    = M.M_mx_FT(w,mu,nu)
Mmx_w    = Mmx_w[0] + 1.j*Mmx_w[1]
# McCarthy:
''' # do once - takes ages
Jmc_w = np.zeros(nf,dtype=complex)
for i in range(nf):
  win = w[i]
  J1,J2 = M.Jf(win,tM,Ju)
  Jmc_w[i] = J1-1.j*J2
Mmc_w = 1./Jmc_w

mout = np.zeros((nf,3))
mout[:,0] = w
mout[:,1] = np.real(Mmc_w)
mout[:,2] = np.imag(Mmc_w)
np.savetxt('yt_true.dat',mout,fmt='%e')
'''

data = np.loadtxt('yt_true.dat')
Mmc_w = data[:,1] + 1.j*data[:,2]


Mw_0,Ms_0   = M.makegxm(s,w3,t3)

# Main McCarthy (N=10)
Mw_10,Ms_10  = M.makegxm(s,w1,t1)
# Supp McCarthy (N=20)
Mw_20,Ms_20  = M.makegxm(s,w2,t2)
'''
plt.figure(1,figsize=(4,2.5))
plt.semilogx(s,Mmx_s,'k--')
plt.semilogx(s,Ms_0,'k-',lw=7)
plt.semilogx(s,Ms_20,'-',color='orange',lw=4)
plt.semilogx(s,Ms_10,'-',color='tab:blue',lw=2)
plt.ylabel(r'$M$')
plt.xlabel(r'$s$')

plt.legend(['Maxwell','True YT',r'$N=20$',r'$N=10$'],\
  frameon=False,loc='lower right')
plt.semilogx([1,1],[-5,5],'k:',lw=1,zorder=0)
plt.xticks([1e-8,1e-0,1e8,1e16])
plt.ylim([-0.05,1.1])
plt.yticks([0,1])
plt.xlim([1e-8,1e16])

plt.tight_layout()

plt.savefig('Mfit.pdf')
'''

plt.figure(2,figsize=(5,5))

plt.subplot(2,1,1)
plt.semilogx(w,np.real(Mmx_w),'k--')
plt.semilogx(w,np.real(Mmc_w),'k-',lw=3)
plt.semilogx(w,np.real(Mw_20),'-',color='orange',lw=2)
plt.semilogx(w,np.real(Mw_10),'-',color='tab:blue',lw=2)
plt.ylabel(r'Re$[\bar{M}]$')
plt.ylim([-0.05,1.05])
plt.yticks([])

plt.legend(['Maxwell','True MC',r'$N=20$',r'$N=10$'],\
  frameon=False,loc='lower right')
plt.semilogx([1,1],[-5,5],'k:',lw=1,zorder=0)
plt.xlim([1e-8,1e16])
plt.xticks([])

plt.subplot(2,1,2)
plt.semilogx(w,np.imag(Mmx_w),'k--')
plt.semilogx(w,np.imag(Mmc_w),'k-',lw=3)
plt.semilogx(w,np.imag(Mw_20),'-',color='orange',lw=2)
plt.semilogx(w,np.imag(Mw_10),'-',color='tab:blue',lw=2)
plt.ylabel(r'Im$[\bar{M}]$')
plt.xlabel(r'$\bar{\omega}$')

plt.yticks([])
plt.ylim([-0.05,0.6])
plt.semilogx([1,1],[-5,5],'k:',lw=1,zorder=0)
plt.xlim([1e-8,1e16])
plt.xticks([1e-8,1e-0,1e8,1e16])

plt.tight_layout()

plt.savefig('Mfit.pdf')
plt.show()
