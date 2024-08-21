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

ds=xr.open_mfdataset("../data/phiinxy*.nc")
print(ds)
phi=da.array(ds.phi)
dns=da.array(ds.dns)
omg=da.array(ds.omg)
x=(ds.x).to_numpy()
y=(ds.y).to_numpy()
t=(ds.t).to_numpy()
nx=len(x)
ny=len(y)
nt=len(t)

print(t.shape)   # Time: t
print(y.shape)   # Coordinate: y
print(x.shape)   # Coordinate: x
print(dns.shape) # Density: n
print(omg.shape) # Vorticisy: omega = \nabla^2 phi
print(phi.shape) # Electrostatic potential: phi


# In[ ]:


### Example of 2D color contour plot ###
fig = plt.figure()
ax = fig.add_subplot(111)
quad = ax.pcolormesh(x,y,phi[0,:,:],
                     shading="auto",cmap="plasma")
fig.colorbar(quad,shrink=1.0,aspect=5)
plt.show()


# In[ ]:


# ### Example of animation ###
# from matplotlib.animation import FuncAnimation
# from IPython.display import HTML

# x2, y2 = np.meshgrid(x,y)
# fig = plt.figure()

# def y2_labframe(time,x2,y2):
#     if gammae == 0:
#         y2_labframe = y2
#     else:
#         dt_remap = ly / lx / gammae
#         num_remap = int((time + 0.5*dt_remap - 0.001)/dt_remap)
#         y2_labframe = y2 + x2 * (gammae * time - num_remap*ly/lx) 
#     return y2_labframe

# def update_quad(i):
#     plt.clf()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel("x_labframe")
#     ax.set_ylabel("y_labframe")
#     ax.set_ylim(-ly,ly)
#     title=fig.suptitle(r"$\phi(x,y)$ Time = {:5.2f}".format(t[i]))
#     quad=ax.pcolormesh(x2, y2_labframe(t[i],x2,y2), phi[i,:,:],
#                        shading="auto",cmap="jet")
# # ### For periodic copy ###
# #     quad=ax.pcolormesh(x2[1:,:], y2_labframe(t[i],x2,y2)[1:,:]+2*ly, phi[i,1:,:],
# #                        shading="auto",cmap="jet")
# #     quad=ax.pcolormesh(x2[:-1,:], y2_labframe(t[i],x2,y2)[:-1,:]-2*ly, phi[i,:-1,:],
# #                        shading="auto",cmap="jet")
# # #########################
#     vmax=np.max(np.abs(phi[i,:,:]))
#     quad.set_clim(-vmax,vmax)
#     cbar=fig.colorbar(quad,shrink=1.0,aspect=5)
    
# ani = FuncAnimation(fig, update_quad,
#                     frames=range(0,len(t),10), interval=100)
# #ani.save('advection.mp4', writer="ffmpeg", dpi=100)

# #plt.show()
# HTML(ani.to_jshtml())


# In[ ]:


# ### Example of animation ###
# from matplotlib.animation import FuncAnimation
# from IPython.display import HTML

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel("x_movingframe")
# ax.set_ylabel("y_movingframe")
# title=fig.suptitle(r"$\phi(x,y)$ Time = {:5.2f}".format(t[0]))
# quad=ax.pcolormesh(x, y, phi[0,:,:],
#                    shading="auto",cmap="jet")
# vmax=np.max(np.abs(phi[0,:,:]))
# quad.set_clim(-vmax,vmax)
# cbar=fig.colorbar(quad,shrink=1.0,aspect=5)

# def update_quad(i):
#     title.set_text(r"$\phi(x,y)$ Time = {:5.2f}".format(t[i]))
#     quad.set_array(np.array([phi[i,:,:]]).flatten())
#     vmax=np.max(np.abs(phi[i,:,:]))
#     quad.set_clim(-vmax,vmax)
    
# ani = FuncAnimation(fig, update_quad,
#                     frames=range(0,len(t),10), interval=100)
# #ani.save('advection.mp4', dpi=100)

# #plt.show()
# HTML(ani.to_jshtml())


# In[ ]:


### Example of animation ###
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("y")
title=fig.suptitle(r"$\phi(x,y)$ Time = {:5.2f}".format(t[0]))
quad=ax.pcolormesh(x, y, phi[0,:,:], shading="auto",cmap="jet")
vmax=np.max(np.abs(phi[0,:,:]))
quad.set_clim(-vmax,vmax)
cbar=fig.colorbar(quad,shrink=1.0,aspect=5)

def update_quad(i):
    title.set_text(r"$\phi(x,y)$ Time = {:5.2f}".format(t[i]))
    quad.set_array(np.array([phi[i,:,:]]).flatten())
    vmax=np.max(np.abs(phi[i,:,:]))
    quad.set_clim(-vmax,vmax)

for i in range(0,len(t),1):
    update_quad(i)
    fig.savefig("./png_phiinxy/phiinxy_t{:08d}".format(i),dpi=100)

plt.show()


# In[ ]:





# In[ ]:




