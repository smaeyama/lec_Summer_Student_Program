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
phi=np.array(ds.rephi + 1j*ds.imphi)
dns=np.array(ds.redns + 1j*ds.imdns)
omg=np.array(ds.reomg + 1j*ds.imomg)
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


### Example of multiple line plot ###
kx2, ky2 = np.meshgrid(kx,ky)
ksq = kx2**2+ky2**2
filt = np.ones_like(ksq)
filt[0,0:nkx] = 0.0
enetky = np.sum(filt.reshape(1,nky+1,2*nkx+1) * ksq.reshape(1,nky+1,2*nkx+1) * abs(phi)**2, axis=2)

fig = plt.figure()
ax = fig.add_subplot(111)
for my in range(nky+1):
    ax.plot(t,enetky[:,my],label=r"$k_y$={:5.2f}".format(ky[my]))
ax.set_yscale("log")
vmax=float(np.max(enetky))
print(abs(phi[10:]).max())
ax.set_ylim(vmax*1e-8,vmax)
ax.legend(bbox_to_anchor=(1,0),loc="lower left",ncol=3)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




