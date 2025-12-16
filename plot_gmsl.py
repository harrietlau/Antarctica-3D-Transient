#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

ff32   = 'GMSL/GMSL_F32.dat'
fNCR   = 'GMSL/GMSL_NCR.dat'
fexp05 = 'GMSL/GMSL_exp05.dat'
fexpA5 = 'GMSL/GMSL_expA5.dat'

[t1,s1] = np.loadtxt(ff32,unpack=True)
[t2,s2] = np.loadtxt(fNCR,unpack=True)
[t3,s3] = np.loadtxt(fexp05,unpack=True)
[t4,s4] = np.loadtxt(fexpA5,unpack=True)

plt.figure(1)
plt.plot(t1,s1,'r-')
plt.plot(t2,s2,'b-')
plt.plot(t3,s3,'k-')
plt.plot(t4,s4,'g-')
plt.xlabel('y')
plt.ylabel('GMSL (m)')
plt.tight_layout()
plt.show()
