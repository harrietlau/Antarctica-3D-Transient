'''
 All functions to test the creep-relaxation
 relations in creep-relax.py
'''

import numpy as np
import scipy.interpolate as si

# --------------------------------------------- #
# idealized models:
# --------------------------------------------- #

def M_mx_FT(w,mu1,nu1):
  # M1 and M2 in Fourier transform
  # domain (w).
  t  = nu1/mu1
  M  = 1.j*w*nu1/(1.+1.j*w*t)
  M1 = np.real(M)
  M2 = np.imag(M)

  return M1,M2

def M_mx_LT(s,mu1,nu1):
  # M in Laplace transform
  # domain (s)
  M  = mu1*s/(s+mu1/nu1)
    
  return M

def M_bg_FT(w,mu1,mu2,nu1,nu2):
  # M1 and M2 in Fourier transform
  # domain (w)
  t1 = nu1/mu1
  t2 = nu2/mu2
  t  = nu1/mu2
  ts = t + t1 + t2
  M  = 1.j*w*nu1*(1.+1.j*w*t2)/ \
       (1.+1.j*w*(ts+1.j*w*t1*t2))
  M1 = np.real(M)
  M2 = np.imag(M)

  return M1,M2

def M_bg_LT(s,mu1,mu2,nu1,nu2):
  # M in Laplace transform
  # domain (s)

  M  = mu1*s*(s+mu2/nu2)/ \
       (s*s + ( (mu1+mu2)/nu2 + mu1/nu1 )*s + \
       mu1*mu2/(nu1*nu2))

  return M

def Mt_mx(nu,mu,t):
  tm = nu/mu
  Mt = mu*np.exp(-t/tm)
  return Mt


def Mt_gmx(wgts,taus,t):
  n  = len(wgts)
  nt = len(t)
  Mt = np.zeros(nt) 
  for i in range(n):
    Mt = Mt + wgts[i]*np.exp(-t/taus[i])
  return Mt
                             


  

# --------------------------------------------- #
# McCarthy
# --------------------------------------------- #

def intgrdX1(taun):
    # Eq. 25, taun< 10^-11
    intgX = 1853.*taun**0.5
    intgX = intgX/taun
    return intgX        

def intgrdX2(taun):
    # Eq. 25, taun>=10^-11
    intgX = 0.32*taun**\
        (0.39-0.28/(1.+2.6*taun**0.1))
    intgX = intgX/taun
    return intgX


def Jf(w,tM,Ju):
  # J in the Fourier transform domain

  fn = w/(2.*np.pi) * tM
  tn = (2.*np.pi*fn)**(-1.)
  
  # get integrated relaxation
  # after many tests, we do this in
  # linear space
  tcut = 1e-11
  ts  = np.linspace(0,tn,2000000)
  dt  = ts[1]-ts[0]
  ts  = np.concatenate((ts,[tn+dt]))
  ts  = ts + dt 
  i0  = np.where(ts< tcut)[0]
  i1  = np.where(ts>=tcut)[0]
  y   = np.zeros(2000001)
  y[i0] = intgrdX1(ts[i0])
  y[i1] = intgrdX2(ts[i1])
  dts = np.diff(ts)
  y   = y[0:-1]
  intX = np.sum(y*dts)
  Xn   = y[-1]*tn
  J1 = Ju*(1. + intX)
  J2 = Ju*(0.5*np.pi*Xn + 1./(w*tM))
    
  return J1,J2


# --------------------------------------------- #
# Fitting
# --------------------------------------------- #

def makeA(w,tau):
  m = len(w)
  n = len(tau)
  A = np.zeros((2*m,n))

  jj = 0
  for j in range(m):
    
    # real:
    for i in range(n):
      den = 1.+w[j]*w[j]*tau[i]*tau[i]
      num = w[j]*w[j]*tau[i]*tau[i]
      A[jj,i] = num/den

    jj = jj + 1
    # imag:
    for i in range(n):
      den = 1.+w[j]*w[j]*tau[i]*tau[i]
      num = w[j]*tau[i]
      A[jj,i] = num/den

    jj = jj + 1

  return A

def makegxm(fs,wgts,taus):
  nf = len(fs)
  n  = len(wgts)

  Mw = np.zeros(nf,dtype=complex)
  Ms = np.zeros(nf)

  for j in range(nf):
    for i in range(n):
      den   = 1.+(fs[j]*fs[j]*taus[i]*taus[i])
      numr  = wgts[i]*taus[i]*taus[i]*fs[j]*fs[j]
      numi  = wgts[i]*taus[i]*fs[j]
      Mw[j] = Mw[j] + (numr + 1.j*numi)/den

      den   = 1.+fs[j]*taus[i]
      num   = wgts[i]*taus[i]*fs[j]
      Ms[j] = Ms[j] + num/den

  return Mw,Ms
