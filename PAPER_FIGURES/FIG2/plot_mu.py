#! /usr/bin/env python3

'''
  Plot mu for schematic diagrams.
  
  See get_mu.py for information.
  
'''

import numpy as np
import matplotlib.pyplot as plt

[s,mx0,mxc,mxh] = np.loadtxt('mu_mx_s.dat',\
  unpack=True,skiprows=1)

[s,mc0,mcc,mch] = np.loadtxt('mu_mc_s.dat',\
  unpack=True,skiprows=1)

tm0 = 1.
tmh = 1e-2
tmc = 1e2

tau = 1./s

plt.figure(1,figsize=(5,3))
plt.semilogx([tm0,tm0],[0,1.3],'k-',lw=1)
plt.semilogx(tau,mx0,'k-',lw=4)
plt.semilogx(tau,mc0,'k--',lw=2)
plt.xlabel(r'log[$\tau$]')
plt.xticks([])
plt.ylabel(r'$M(s)$')
plt.yticks([])
plt.xlim([1e-12,1e12])
plt.ylim([0,1.3])

plt.tight_layout()

plt.savefig('mu0.pdf')

plt.figure(2,figsize=(5,3))
plt.semilogx([tm0,tm0],[0,1.3],'k-',lw=1)
plt.semilogx(tau,mx0,'k-',lw=4)
plt.semilogx(tau,mc0,'k--',lw=2)
plt.semilogx(tau,mxc,'-',lw=2,color='tab:blue')
plt.semilogx(tau,mcc,'--',lw=2,color='tab:blue')
plt.semilogx([tmc,tmc],[0,1.3],'-',lw=1,color='tab:blue')
plt.xlabel(r'log[$\tau$]')
plt.xticks([])
plt.ylabel(r'$M(s)$')
plt.yticks([])
plt.xlim([1e-12,1e12])
plt.ylim([0,1.3])

plt.tight_layout()

plt.savefig('mu_cold.pdf')

plt.figure(3,figsize=(5,3))
plt.semilogx([1./tm0,1./tm0],[0,1.3],'k-',lw=1)
plt.semilogx(tau,mx0,'k-',lw=4)
plt.semilogx(tau,mc0,'k--',lw=2)
plt.semilogx(tau,mxh,'-',lw=2,color='firebrick')
plt.semilogx(tau,mch,'--',lw=2,color='firebrick')
plt.semilogx([tmh,tmh],[0,1.3],'-',lw=1,color='firebrick')
plt.xlabel(r'log[$\tau$]')
plt.xticks([])
plt.ylabel(r'$M(s)$')
plt.yticks([])
plt.xlim([1e-12,1e12])
plt.ylim([0,1.3])

plt.tight_layout()

plt.savefig('mu_hot.pdf')

plt.show()
