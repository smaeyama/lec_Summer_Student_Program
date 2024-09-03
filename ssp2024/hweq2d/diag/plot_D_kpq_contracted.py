#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob

ds = xr.open_dataset("./data_netcdf/D_kpq_1400-1499.nc")
print(ds)
D_kpq = (ds.D_kpq).to_numpy()
ky=(ds.ky).to_numpy()
kx=(ds.kx).to_numpy()
nkx=int((len(kx)-1)/2)
nky=int((len(ky)-1)/2)

## Average over files ###
filelist = sorted(glob.glob('./data_netcdf/D_kpq_*.nc'))
print(filelist)
D_kpq = D_kpq * 0.0
count = 0
for file in filelist[2:-1]:
    ds = xr.open_dataset(file)
    D_kpq = D_kpq + (ds.D_kpq).to_numpy()
    count = count + 1
D_kpq = D_kpq / count
##

D_kpq_max = np.max([abs(D_kpq.max()),abs(D_kpq.min())])
# D_kpq = D_kpq / D_kpq_max
print("D_kpq_max(abs):", D_kpq_max)
# print("D_kpq_max(normalized):",D_kpq.max(),", D_kpq_min(normalized):",D_kpq.min())

print("kxmin=",kx[1],", kymin=",ky[1])


# ### Analyze inverse/forward cascade

# In[ ]:


inverse_D_kpq_giver = np.zeros(D_kpq.shape[0:2])
inverse_D_kpq_taker = np.zeros(D_kpq.shape[0:2])
forward_D_kpq_giver = np.zeros(D_kpq.shape[0:2])
forward_D_kpq_taker = np.zeros(D_kpq.shape[0:2])
kx2, ky2 = np.meshgrid(kx, ky)
ksq = kx2**2 + ky2**2
print(ksq.shape)
for my in range(len(ky)):
    for mx in range(len(kx)):
        ksq_k = ksq[my,mx]
        inverse_D_kpq_giver[my,mx] = np.sum(D_kpq[my,mx,:,:][(ksq <= ksq_k) & (D_kpq[my,mx,:,:] < 0)]) # inverse cascade: the mode k gives energy to low-q modes
        inverse_D_kpq_taker[my,mx] = np.sum(D_kpq[my,mx,:,:][(ksq > ksq_k) & (D_kpq[my,mx,:,:] > 0)])  # inverse cascade: the mode k takes energy from high-q modes
        forward_D_kpq_giver[my,mx] = np.sum(D_kpq[my,mx,:,:][(ksq > ksq_k) & (D_kpq[my,mx,:,:] < 0)])  # forward cascade: the mode k gives energy to high-q modes
        forward_D_kpq_taker[my,mx] = np.sum(D_kpq[my,mx,:,:][(ksq <= ksq_k) & (D_kpq[my,mx,:,:] > 0)]) # forward cascade: the mode k takes energy from low-q modes


T_k = np.sum(D_kpq, axis=(-2,-1))
fig = plt.figure(figsize=(3.5,3))
ax = fig.add_subplot(111)
xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Total transfer $T_k=\sum_p \sum_q D_{k \leftarrow q}^{p}$")
kx_shift = np.fft.fftshift(kx)
ky_shift = np.fft.fftshift(ky)
T_k_shift = np.fft.fftshift(T_k)
vmax = abs(T_k).max()
quad = ax.pcolormesh(kx_shift,ky_shift,T_k_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(10.5,3))
ax = fig.add_subplot(131)
# xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Inverse cascade (give+take)")
# kx_shift = np.fft.fftshift(kx)
# ky_shift = np.fft.fftshift(ky)
inverse_D_kpq_shift = np.fft.fftshift(inverse_D_kpq_giver+inverse_D_kpq_taker)
# vmax = abs(inverse_D_kpq_shift).max()
quad = ax.pcolormesh(kx_shift,ky_shift,inverse_D_kpq_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
ax = fig.add_subplot(132)
# xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Inverse (give) $\sum_p \sum_{q \leq k} D_{k \leftarrow q}^{p}<0$")
# kx_shift = np.fft.fftshift(kx)
# ky_shift = np.fft.fftshift(ky)
inverse_D_kpq_giver_shift = np.fft.fftshift(inverse_D_kpq_giver)
# vmax = abs(inverse_D_kpq_shift).max()
quad = ax.pcolormesh(kx_shift,ky_shift,inverse_D_kpq_giver_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
ax = fig.add_subplot(133)
# xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Inverse (take) $\sum_p \sum_{q>k} D_{k \leftarrow q}^{p}>0$")
# kx_shift = np.fft.fftshift(kx)
# ky_shift = np.fft.fftshift(ky)
inverse_D_kpq_taker_shift = np.fft.fftshift(inverse_D_kpq_taker)
# vmax = abs(inverse_D_kpq_shift).max()
quad = ax.pcolormesh(kx_shift,ky_shift,inverse_D_kpq_taker_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
fig.tight_layout()
plt.show()


fig = plt.figure(figsize=(10.5,3))
ax = fig.add_subplot(131)
# xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Forward cascade (give+take)")
# kx_shift = np.fft.fftshift(kx)
# ky_shift = np.fft.fftshift(ky)
forward_D_kpq_shift = np.fft.fftshift(forward_D_kpq_giver+forward_D_kpq_taker)
# vmax = abs(forward_D_kpq_shift).max()
quad = ax.pcolormesh(kx_shift,ky_shift,forward_D_kpq_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
ax = fig.add_subplot(132)
# xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Forward (give) $\sum_p \sum_{q > k} D_{k \leftarrow q}^{p}<0$")
# kx_shift = np.fft.fftshift(kx)
# ky_shift = np.fft.fftshift(ky)
forward_D_kpq_giver_shift = np.fft.fftshift(forward_D_kpq_giver)
# vmax = abs(forward_D_kpq_shift).max()
quad = ax.pcolormesh(kx_shift,ky_shift,forward_D_kpq_giver_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
ax = fig.add_subplot(133)
# xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Forward (take) $\sum_p \sum_{q \leq k} D_{k \leftarrow q}^{p}>0$")
# kx_shift = np.fft.fftshift(kx)
# ky_shift = np.fft.fftshift(ky)
forward_D_kpq_taker_shift = np.fft.fftshift(forward_D_kpq_taker)
# vmax = abs(forward_D_kpq_shift).max()
quad = ax.pcolormesh(kx_shift,ky_shift,forward_D_kpq_taker_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
fig.tight_layout()
plt.show()


fig = plt.figure(figsize=(3.5,3))
ax = fig.add_subplot(111)
# xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Check difference")
quad = ax.pcolormesh(kx_shift,ky_shift,T_k_shift - (inverse_D_kpq_shift + forward_D_kpq_shift),
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
plt.show()


# ### Analyze transfer between kx <- qx

# In[ ]:


contracted_D_kxqx = np.sum(D_kpq, axis=(0,2)) # contracted_D_kxqx[kx,qx]
print(D_kpq.shape,contracted_D_kxqx.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$q_x$")
ax.set_ylabel(r"$k_x$")
ax.set_aspect("equal")
ax.set_title(r"Contracted $D_{k_x \leftarrow q_x} = \sum_{k_y} \sum_{q_y} \sum_p D_{k \leftarrow q}^p$")
kx_shift = np.fft.fftshift(kx)
contracted_D_kxqx_shift = np.fft.fftshift(contracted_D_kxqx)
vmax = abs(contracted_D_kxqx_shift).max()
quad = ax.pcolormesh(kx_shift,kx_shift,contracted_D_kxqx_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
plt.show()


# ### Analyze transfer between ky <- qy

# In[ ]:


contracted_D_kyqy = np.sum(D_kpq, axis=(1,3)) # contracted_D_kyqy[ky,qy]
print(D_kpq.shape,contracted_D_kyqy.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$q_y$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect("equal")
ax.set_title(r"Contracted $D_{k_y \leftarrow q_y} = \sum_{k_x} \sum_{q_x} \sum_p D_{k \leftarrow q}^p$")
ky_shift = np.fft.fftshift(ky)
contracted_D_kyqy_shift = np.fft.fftshift(contracted_D_kyqy)
vmax = abs(contracted_D_kyqy_shift).max()
quad = ax.pcolormesh(ky_shift,ky_shift,contracted_D_kyqy_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
fig.colorbar(quad)
plt.show()


# In[ ]:




