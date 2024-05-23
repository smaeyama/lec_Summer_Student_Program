#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import dask.array as da
import f90nml

nml = f90nml.read("../param.namelist")
nx = nml["numer"]["nx"]
ny = nml["numer"]["ny"]
lx = nml["numer"]["lx"]
ly = nml["numer"]["ly"]
ca = nml["physp"]["ca"]
nu = nml["physp"]["nu"]
eta = nml["physp"]["eta"]
gammae = nml["physp"]["gammae"]
init_ampl = nml["physp"]["init_ampl"]
print("# ca =",ca)
print("# nu =",nu)
print("# eta =",eta)
print("# gammae =",gammae)
print("# init_ampl =",init_ampl)

ds=xr.open_mfdataset("../data/phiinkxky*.nc")
print(ds)
phi=da.array(ds.rephi + 1j*ds.imphi)
dns=da.array(ds.redns + 1j*ds.imdns)
omg=da.array(ds.reomg + 1j*ds.imomg)
kx=np.array(ds.kx)
ky=np.array(ds.ky)
t=np.array(ds.t)
nkx=int((len(kx)-1)/2)
nky=len(ky)-1
nt=len(t)

print(t.shape)   # Time: t
print(ky.shape)  # Coordinate: ky
print(kx.shape)  # Coordinate: kx
print(dns.shape) # Density: n
print(omg.shape) # Vorticisy: omega = \nabla^2 phi
print(phi.shape) # Electrostatic potential: phi


# In[ ]:


def copy_extended_k_space(nkx,nky,phi):
    phik = np.zeros([phi.shape[0],2*nky+1,2*nkx+1],dtype=np.complex128)
    phik[:,0:nky+1,0:nkx+1] = phi[:,0:nky+1,nkx:2*nkx+1]
    phik[:,0:nky+1,nkx+1:2*nkx+1] = phi[:,0:nky+1, 0:nkx]
    phik[:,nky+1:2*nky+1,nkx+1:2*nkx+1] = np.conj(phi[:,nky:0:-1,2*nkx:nkx:-1])
    phik[:,nky+1:2*nky+1,0:nkx+1] = np.conj(phi[:,nky:0:-1,nkx::-1])
    return phik

kx_shift = np.zeros([2*nkx+1])
kx_shift[0:nkx+1] = kx[nkx:]
kx_shift[-nkx:] = kx[:nkx]
ky_shift = np.zeros([2*nky+1])
ky_shift[0:nky+1] = ky[:]
ky_shift[-nky:] = -ky[nky:0:-1]
kx2, ky2 = np.meshgrid(kx_shift, ky_shift)
ksq = kx2**2 + ky2**2
phik = copy_extended_k_space(nkx,nky,phi)
omgk = copy_extended_k_space(nkx,nky,omg)
dnsk = copy_extended_k_space(nkx,nky,dns)


# In[ ]:


def calc_pbk_k(mx,my,nkx,nky,kx,ky,fk,gk):
    pbk = np.zeros(fk.shape[0],dtype=np.complex128)
    for py in range(max(-nky-my, -nky), min(nky, nky-my)+1):
        qy = -py-my
        for px in range(max(-nkx-mx, -nkx), min(nkx, nkx-mx)+1):
            qx = -px-mx
            wkpbk = - (kx[px]*ky[qy]-ky[py]*kx[qx])*fk[:,py,px]*gk[:,qy,qx]
            pbk = pbk + wkpbk
    pbk = np.conjugate(pbk)
    return pbk

# # Check: calc_pbk_k
# mx = 2
# my = -8
# wk_t = (t[:-1]+t[1:])/2
# domgkdt = np.diff(omgk[:,my,mx]) / (t[1]-t[0])
# pbk_phiomg = calc_pbk_k(mx,my,nkx,nky,kx_shift,ky_shift,phik,omgk)
# rhs = - pbk_phiomg - ca * ky2[my,mx]**2 * (dnsk[:,my,mx] - phik[:,my,mx]) - nu * ksq[my,mx]**2 * omgk[:,my,mx]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(wk_t,domgkdt.real)
# ax.plot(t,rhs.real)
# ax.plot(wk_t,domgkdt.imag)
# ax.plot(t,rhs.imag)
# ax.set_title(r"kx={:5.2f},ky={:5.2f}".format(kx_shift[mx],ky_shift[my]))
# plt.show()


# In[ ]:


def calc_Skpq_k(mx,my,nkx,nky,kx,ky,fk,gk,hk):
    Skpq = np.zeros([fk.shape[0],2*nky+1,2*nkx+1],dtype=np.float64)
    for py in range(max(-nky-my, -nky), min(nky, nky-my)+1):
        qy = -py-my
        for px in range(max(-nkx-mx, -nkx), min(nkx, nkx-mx)+1):
            qx = -px-mx
            Skpq[:,py,px] = - 0.5 * (kx[px]*ky[qy]-ky[py]*kx[qx]) * ((fk[:,py,px]*gk[:,qy,qx] - fk[:,qy,qx]*gk[:,py,px])*hk[:,my,mx]).real
    return Skpq

# # Check: calc_Skpq_k
# mx = 0
# my = 10
# wk_t = (t[:-1]+t[1:])/2
# domgkdt = np.diff(omgk[:,my,mx]) / (t[1]-t[0])
# S_kpq_phiomg = calc_Skpq_k(mx,my,nkx,nky,kx_shift,ky_shift,phik,omgk,phik)
# T_kpq_phiomg = np.sum(S_kpq_phiomg, axis=(1,2))
# pbk_phiomg = calc_pbk_k(mx,my,nkx,nky,kx_shift,ky_shift,phik,omgk)
# rhs = np.real(np.conjugate(-phik[:,my,mx]) * (- pbk_phiomg))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(t,T_kpq_phiomg)
# ax.plot(t,rhs)
# ax.set_title(r"kx={:5.2f},ky={:5.2f}".format(kx_shift[mx],ky_shift[my]))
# plt.show()


# In[ ]:


from time import time as timer
from numba import njit, prange

@njit(parallel=True)
def calc_Skpq_time_averaged(itsta,itend,nkx,nky,kx,ky,fk,gk,hk):
    Skpq = np.zeros((2*nky+1,2*nkx+1,2*nky+1,2*nkx+1),dtype=np.float64)
    for my in prange(-nky,nky+1): # Parallelization by Numba
    # for my in range(-nky,nky+1):
        for py in range(max(-nky-my, -nky), min(nky, nky-my)+1):
            qy = -py-my
            for mx in range(-nkx,nkx+1):
                for px in range(max(-nkx-mx, -nkx), min(nkx, nkx-mx)+1):
                    qx = -px-mx
                    fgh_ave = np.average(((fk[itsta:itend,py,px]*gk[itsta:itend,qy,qx] - fk[itsta:itend,qy,qx]*gk[itsta:itend,py,px])*hk[itsta:itend,my,mx]).real)
                    Skpq[my,mx,py,px] = - 0.5 * (kx[px]*ky[qy]-ky[py]*kx[qx]) * fgh_ave
    return Skpq

# Check: calc_Skpq_time_averaged
mx = 0
my = 10
S_kpq_phiomg = calc_Skpq_k(mx,my,nkx,nky,kx_shift,ky_shift,phik,omgk,phik)

itsta = 100
itend = 200

t1 = timer()
Skpq_ave = calc_Skpq_time_averaged(itsta,itend,nkx,nky,kx_shift,ky_shift,phik,omgk,phik)
t2 = timer(); print("Elapsed time [s]:", t2-t1)

print(t[itsta],t[itend])
Skpq_ave = calc_Skpq_time_averaged(itsta,itend,nkx,nky,kx_shift,ky_shift,phik,omgk,phik)
fig = plt.figure(figsize=(10,4))
vmax = np.max([Skpq_ave[my,mx,:,:].max(),-Skpq_ave[my,mx,:,:].min()])
ax = fig.add_subplot(131)
ax.pcolormesh(np.fft.fftshift(kx2),np.fft.fftshift(ky2),np.fft.fftshift(Skpq_ave[my,mx,:,:]),vmax=vmax,vmin=-vmax,cmap="jet")
ax = fig.add_subplot(132)
ax.pcolormesh(np.fft.fftshift(kx2),np.fft.fftshift(ky2),np.fft.fftshift(np.average(S_kpq_phiomg[itsta:itend,:,:],axis=0)),vmax=vmax,vmin=-vmax,cmap="jet")
ax = fig.add_subplot(133)
ax.pcolormesh(np.fft.fftshift(kx2),np.fft.fftshift(ky2),np.fft.fftshift(Skpq_ave[my,mx,:,:]-np.average(S_kpq_phiomg[itsta:itend,:,:],axis=0)),vmax=vmax,vmin=-vmax,cmap="seismic")
plt.show()


# In[ ]:


chunk = 100
n_out = len(t)
split = int((n_out-1)/chunk)+1
print(n_out,chunk,split)
for i in range(split):
    sta=i*chunk
    end=min((i+1)*chunk,n_out)
    print(i,sta,end)
    Skpq_ave = calc_Skpq_time_averaged(sta,end,nkx,nky,kx_shift,ky_shift,phik,omgk,phik)
    xr_S_kpq=xr.DataArray(Skpq_ave,dims=("ky","kx","qy","qx"),coords={"ky":ky_shift,"kx":kx_shift,"qy":ky_shift,"qx":kx_shift})
    ds=xr.Dataset({"S_kpq":xr_S_kpq}, 
                  attrs={"description":"S_kpq is the symmetrized energy transfer function S_k^pq. \n"+
                                       "S_kpq means energy gain (S>0) or loss (S<0) of the mode k via the coupling with modes p and q.\n"+
                                       "    Fourier mode coupling condition: k+p+q=0. \n"+
                                       "    Symmetry: S_k^pq = S_k^qp. \n"+
                                       "    Detailed balance: S_k^pq+S_p^qk+S_q^kp=0. \n"+
                                       "    Relation to net energy gain of the mode k: T_k = \sum_p \sum_q S_k^pq.",
                        "time-window":"Averaged over {:}<=t<={:}".format(t[sta],t[end-1])})
    ds.to_netcdf("./data_netcdf/S_kpq_{:04d}-{:04d}.nc".format(sta,end-1),mode="w")


# In[ ]:


ds = xr.open_dataset("./data_netcdf/S_kpq_0000-0099.nc")
print(ds)


# In[ ]:




