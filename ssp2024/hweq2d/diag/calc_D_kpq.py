#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import dask.array as da
import f90nml
import glob

filelist = sorted(glob.glob("./data_netcdf/S_kpq*nc"))
print(filelist)


# In[ ]:


from numba import njit, prange
from time import time as timer

@njit(parallel=True)                # Acceleration by Numba, with parallelization at prange
def calculate_D_kpq(S_kpq):
    """
    Directional representation of symmetric triad transfer, D_{k<-q}^p
    D_{k<-q}^p represents symmetric triad transfer as directional transfer from q to k via tha coupling with p.
    
    
    Parameters
    ----------
    S_kpq : Numpy array
        Symetric triad transfer function S_k^pq
        Its shape is (2*nky+1,2*nkx+1,2*nqy+1,2*nqx+1),
        where nky=npy=nqy and nkx=npy=nqy are the numbers of modes in 2D Fourier space.
        Dependence on the mode p is omitted, since p=-k-q from Kronecker's delta in Fourier analysis.
            
    Returns
    -------
    D_kpq : Numpy array
        Directional representation D_{k<-q}^p
        Its shape is (2*nky+1,2*nkx+1,2*nqy+1,2*nqx+1)
        Dependence on the mode p is omitted, since p=-k-q from Kronecker's delta in Fourier analysis.
    
    Theory
    ------
    Because of detailed balance of symmetric triad transfer, S_k^pq + S_p^qk + S_q^kp = 0,
    the interactions among (k,p,q) are always 2 giver(+) - 1 taker(-) or 1 giver(+) - 2 taker(-).
    
    The directional representation splits the symmetric triad transfer according to 
    the rule of "No simultanous gain/loss", or equivalently, 
    "Giver should give, taker should take", or "Minimize |D_{k<-q}^p|+|D_p^qk|+|D_q^kp|".
    
    * No simulaneous gain/loss rule
        If sign(S_k^pq) == sign(S_q^kp), which means both k and p are givers (or takers), then set
        D_{k<-q}^p = 0
    * Conservation law between two (k,q) via a mediator (p)
        D_{k<-q}^p = - D_{q<-k}^p
        
    * Relation with symmetric triad transfer
        S_k^pq = 0.5 * (D_{k<-q}^p + D_{k<-p}^q)
        S_p^qk = 0.5 * (D_{p<-k}^q + D_{p<-q}^k)
        S_q^kp = 0.5 * (D_{q<-p}^k + D_{q<-k}^p)
    """
    D_kpq = np.zeros_like(S_kpq)
    for my in prange(-nky,nky+1):  # Parallelization by Numba
    #for my in range(-nky,nky+1):
        for mx in range(-nkx,nkx+1):
            for qy in range(-nky,nky+1):
                py = -my-qy
                if (abs(py)<=nky):
                    for qx in range(-nkx,nkx+1):
                        px = -mx-qx
                        if (abs(px)<=nkx):
                            wj=np.array([S_kpq[my,mx,qy,qx],S_kpq[py,px,my,mx],S_kpq[qy,qx,py,px]])
                            if ((my,mx)==(py,px) and (py,px)==(qy,qx)): # S_k^kk = 0
                                D_kpq[my,mx,qy,qx] = 0.0
                            elif ((my,mx)==(py,px) or (py,px)==(qy,qx) or (qy,qx)==(my,mx)): # S_k^qq = -2*S_q^kq
                                if (np.prod(wj) > 0):  ## 2 givers(-) and 1 taker(+)
                                    arg=np.argmax(wj)
                                    if arg==0:         ### k is the taker, p=q are givers
                                        D_kpq[my,mx,qy,qx] = S_kpq[my,mx,qy,qx] # Giver -> Taker (posivite value)
                                    elif arg==1:       ### p is the taker, q=k are givers
                                        D_kpq[my,mx,qy,qx] = 0.0 # Giver should only give.
                                    else:              ### q is the taker, k=p are givers
                                        D_kpq[my,mx,qy,qx] = 2*S_kpq[my,mx,qy,qx] # Taker -> Giver (negative value)
                                else:                  ## 1 giver(-) and 2 takers(+)
                                    arg=np.argmin(wj)
                                    if arg==0:         ### k is the giver, p=q are takers
                                        D_kpq[my,mx,qy,qx] = S_kpq[my,mx,qy,qx] # Taker -> Giver (negative value)
                                    elif arg==1:       ### p is the giver, q=k are takers
                                        D_kpq[my,mx,qy,qx] = 0.0 # Taker should only take.
                                    else:              ### q is the giver, k=p are takers
                                        D_kpq[my,mx,qy,qx] = 2*S_kpq[my,mx,qy,qx] # Giver -> Taker (posivite value)
                            else:                      # S_k^pq + S_p^qk + S_q^kp = 0
                                if (np.prod(wj) > 0):  ## 2 givers(-) and 1 taker(+)
                                    arg=np.argmax(wj)
                                    if arg==0:         ### k is the taker, p,q are givers
                                        D_kpq[my,mx,qy,qx] = - 2*S_kpq[qy,qx,my,mx] # Giver -> Taker (posivite value)
                                    elif arg==1:       ### p is the taker, q,k are givers
                                        D_kpq[my,mx,qy,qx] = 0.0 # Giver should only give.
                                    else:              ### q is the taker, k,p are givers
                                        D_kpq[my,mx,qy,qx] = 2*S_kpq[my,mx,qy,qx] # Taker -> Giver (negative value)
                                else:                  ## 1 giver(-) and 2 takers(+)
                                    arg=np.argmin(wj)
                                    if arg==0:         ### k is the giver, p,q are takers
                                        D_kpq[my,mx,qy,qx] = - 2*S_kpq[qy,qx,my,mx] # Taker -> Giver (negative value)
                                    elif arg==1:       ### p is the giver, q,k are takers
                                        D_kpq[my,mx,qy,qx] = 0.0 # Taker should only take.
                                    else:              ### q is the giver, k,p are takers
                                        D_kpq[my,mx,qy,qx] = 2*S_kpq[my,mx,qy,qx] # Giver -> Taker (posivite value)
                                        #print(D_kpq[my,mx,qy,qx],2*wj)
    return D_kpq


# In[ ]:


for f in filelist:
    ds = xr.open_dataset(f)
    S_kpq = np.array(ds.S_kpq)
    ky=np.array(ds.ky)
    kx=np.array(ds.kx)
    nkx=int((len(kx)-1)/2)
    nky=int((len(ky)-1)/2)

    t1 = timer()
    D_kpq = calculate_D_kpq(S_kpq)
    t2 = timer(); print("Elapsed time [sec]:",t2-t1)
    
    xr_D_kpq=xr.DataArray(D_kpq,dims=("ky","kx","qy","qx"),coords={"ky":ky,"kx":kx,"qy":ky,"qx":kx})
    ds=xr.Dataset({"D_kpq":xr_D_kpq},
                  attrs={"description":"D_kpq is the directional representation of triad energy transfer D_{k<-q}^p. \n"+
                                       "D_kpq means energy gain (D>0) or loss (S<0) of the mode k from the mode q via the coupling with the mediator p. \n"+
                                       "    Fourier mode coupling condition: k+p+q=0. \n"+
                                       "    Anti-symmetry: D_{k<-q}^p = -D_{q<-k}^p. \n"+
                                       "    Relation to net energy gain of the mode k: T_k = \sum_p \sum_q D_k^pq. \n"+
                                       "For details, see, S. Maeyama et al., New J. Phys. 23, 043049 (2021). https://doi.org/10.1088/1367-2630/abeffc"})
    outfile = f.replace("S_kpq_","D_kpq_")
    print(outfile)
    ds.to_netcdf(outfile,mode="w")


# In[ ]:


### Check total transfer
T_k = np.sum(S_kpq, axis=(-2,-1))
fig = plt.figure()
ax = fig.add_subplot(111)
xylim=4
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"$T_k=\sum_p \sum_q S_k^{p,q}$")
kx_shift = np.fft.fftshift(kx)
ky_shift = np.fft.fftshift(ky)
T_k_shift = np.fft.fftshift(T_k)
vmax = abs(T_k).max()
quad = ax.pcolormesh(kx_shift,ky_shift,T_k_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
plt.show()

T_k = np.sum(D_kpq, axis=(-2,-1))
fig = plt.figure()
ax = fig.add_subplot(111)
xylim=4
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"$T_k=\sum_p \sum_q D_{k \leftarrow q}^{p}$")
kx_shift = np.fft.fftshift(kx)
ky_shift = np.fft.fftshift(ky)
T_k_shift = np.fft.fftshift(T_k)
vmax = abs(T_k).max()
quad = ax.pcolormesh(kx_shift,ky_shift,T_k_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
plt.show()


# In[ ]:




