#! /usr/bin/env python3

''' 
  Plot Re [M], Im [M] and M(s)
  fits

'''

import numpy as np
import mfunc as M
import scipy.linalg as sl

# ============================================== #
# (1) define some testing paramters/variables
# ============================================== #

# make 3 models for plotting:
# one with 1 (done)
# one with 1.2
# one with 0.8

mcold = 1.2
tcold = 1e2

mhot  = 0.9
thot  = 1e-2


nf  = 500
s   = np.logspace(-20,20,nf)
w   = np.logspace(-20,20,nf)
t   = 1./s

# get M anf J for McCarthy and Maxwell

tM  = 1.
mu  = 1.
Ju  = 1./mu
nu  = 1.


# Maxwell:
M1,M2 = M.M_mx_FT(w,mu,nu)
Mmx_w = M1 + 1.j*M2
Jmx_w = 1./Mmx_w
Mmx_s = M.M_mx_LT(s,mu,nu)

outmat = np.zeros((nf,4))
outmat[:,0] = s
outmat[:,1] = Mmx_s
ncold  = tcold*mcold
Mmx_s  = M.M_mx_LT(s,mcold,ncold)
outmat[:,2] = Mmx_s
nhot   = thot*mhot
Mmx_s = M.M_mx_LT(s,mhot,nhot)
outmat[:,3] = Mmx_s

np.savetxt('mu_mx_s.dat',outmat,fmt='%e',\
  header = 's, Ms for tM=1, tM=1.2, tM=0.8 (Maxwell model)')


# McCarthy:
Jmc_w = np.zeros(nf,dtype=complex)
for i in range(nf):
  win = w[i]
  J1,J2 = M.Jf(win,tM,Ju)
  Jmc_w[i] = J1-1.j*J2
Mmc_w = 1./Jmc_w

# ============================================== #
# (2) Conduct fits                               #
# ============================================== #

# make 'data' vector:
d     = np.zeros(2*nf)
jj = 0
for j in range(nf):
  d[jj] = np.real(Mmc_w[j])
  jj = jj + 1
  d[jj] = np.imag(Mmc_w[j])
  jj = jj + 1

# make testing parameters:
n     = 20
per0  = tM*1e-10
per1  = tM                   # max period is tM
lp0   = np.log10(per0)
lp1   = np.log10(per1)

wgts  = np.zeros(n)
taus  = np.zeros(n)

tau = np.logspace(lp0,lp1,n)
A   = M.makeA(w,tau)
At  = np.transpose(A)
AA  = np.matmul(At,A)
dd  = np.matmul(At,d)
wgt = sl.solve(AA,dd)

# ============================================== #
# (4) Reform model in Laplace domain             #
# ============================================== #

outmat = np.zeros((nf,4))

# (1) tM = 1
Mw,Ms  = M.makegxm(s,wgt,tau)
outmat[:,0] = s
outmat[:,1] = Ms

# (2) tM = 1.2
w1 = wgt*mcold
t1 = tau*tcold
Mw,Ms  = M.makegxm(s,w1,t1)
outmat[:,2] = Ms

# (3) tM = 0.8
w1 = wgt*mhot
t1 = tau*thot
Mw,Ms  = M.makegxm(s,w1,t1)
outmat[:,3] = Ms

np.savetxt('mu_mc_s.dat',outmat,fmt='%e',\
  header = 's, Ms for tM=1, tM=1.2, tM=0.8 (McCarthy model)')

