#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import dask.array as da
import f90nml

nml = f90nml.read("../param.namelist")
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
kx=(ds.kx).to_numpy()
ky=(ds.ky).to_numpy()
t=(ds.t).to_numpy()
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


### Example of 2D color contour plot ###
kx2, ky2 = np.meshgrid(kx,ky)
ksq = kx2**2 + ky2**2
fig = plt.figure()
ax = fig.add_subplot(111)
quad = ax.pcolormesh(kx,ky,np.abs(phi[0,:,:])**2,
                     shading="auto",cmap="plasma")
fig.colorbar(quad,shrink=1.0,aspect=5)
plt.show()


# In[ ]:


# ### Example of animation ###
# from matplotlib.animation import FuncAnimation
# import matplotlib.colors as colors
# from IPython.display import HTML

# kx2, ky2 = np.meshgrid(kx,ky)
# fig = plt.figure()

# def kx2_labframe(time,kx2,ky2):
#     if gammae == 0:
#         kx2_labframe = kx2
#     else:
#         dt_remap = ly / lx / gammae
#         num_remap = int((time + 0.5*dt_remap - 0.001)/dt_remap)
#         kx2_labframe = kx2 - ky2 * (gammae * time - num_remap*ly/lx) 
#     return kx2_labframe

# t0=0
# def update_quad(i):
#     plt.clf()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel("kx_labframe")
#     ax.set_ylabel("ky_labframe")
#     ax.set_xlim(kx[0]-0.5*ly*ky[-1]/lx,kx[-1]+0.5*ly*ky[-1]/lx)
#     title=fig.suptitle(r"$|\omega(k_x,k_y)|$ Time = {:5.2f}".format(t[i]))
#     quad=ax.pcolormesh(kx2_labframe(t[i],kx2,ky2), ky2, np.abs(omg[i,:,:]),
#                        shading="auto",cmap="jet",norm=colors.LogNorm())
#     cbar=fig.colorbar(quad,shrink=1.0,aspect=5)

    
# ani = FuncAnimation(fig, update_quad,
#                     frames=range(int(len(t)*0/3),int(len(t)*3/3),10), interval=100)
# #ani.save('advection.mp4', writer="ffmpeg", dpi=100)

# #plt.show()
# HTML(ani.to_jshtml())


# In[ ]:


# ### Example of animation ###
# from matplotlib.animation import FuncAnimation
# import matplotlib.colors as colors
# from IPython.display import HTML

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel("kx_movingframe")
# ax.set_ylabel("ky_movingframe")
# title=fig.suptitle(r"$|\omega(k_x,k_y)|$ Time = {:5.2f}".format(t[0]))
# quad=ax.pcolormesh(kx, ky, np.abs(omg[0,:,:]),
#                    shading="auto",cmap="jet",norm=colors.LogNorm())
# vmax=np.max(np.abs(omg[0,:,:]))
# quad.set_clim(None,vmax)
# cbar=fig.colorbar(quad,shrink=1.0,aspect=5)

# def update_quad(i):
#     title.set_text(r"$|\omega(k_x,k_y)|$ Time = {:5.2f}".format(t[i]))
#     quad.set_array(np.abs([omg[i,:,:]]).flatten())
#     vmax=np.max(np.abs(omg[i,:,:]))
#     quad.set_clim(None,vmax)
    
# ani = FuncAnimation(fig, update_quad,
#                     frames=range(int(len(t)*0/3),int(len(t)*3/3),10), interval=100)
# #ani.save('advection.mp4', dpi=100)

# #plt.show()
# HTML(ani.to_jshtml())


# In[ ]:


### Example of animation ###
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from IPython.display import HTML

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
title=fig.suptitle(r"$|E(k_x,k_y)|$ Time = {:5.2f}".format(t[0]))
enekxky = ksq * np.abs(phi[0,:,:])**2
quad=ax.pcolormesh(kx, ky, enekxky,
                   shading="auto",cmap="plasma",norm=colors.LogNorm())
vmax=enekxky.max()
quad.set_clim(vmax*1e-4,vmax)
cbar=fig.colorbar(quad,shrink=1.0,aspect=5)

def update_quad(i):
    title.set_text(r"$|E(k_x,k_y)|$ Time = {:5.2f}".format(t[i]))
    enekxky = ksq * np.abs(phi[i,:,:])**2
    quad.set_array(enekxky.flatten())
    vmax=enekxky.max()
    quad.set_clim(vmax*1e-4,vmax)
    
for i in range(0,len(t),10):
    update_quad(i)
    fig.savefig("./png_phiinkxky/phiinkxky_t{:08d}".format(i),dpi=100)

plt.show()


# In[ ]:




