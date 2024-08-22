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
for file in filelist[2:]:
    ds = xr.open_dataset(file)
    D_kpq = D_kpq + (ds.D_kpq).to_numpy()
    count = count + 1
D_kpq = D_kpq / count
##

D_kpq_max = np.max([abs(D_kpq.max()),abs(D_kpq.min())])
D_kpq = D_kpq / D_kpq_max
print("D_kpq_max(abs):", D_kpq_max)
print("D_kpq_max(normalized):",D_kpq.max(),", D_kpq_min(normalized):",D_kpq.min())

print("kxmin=",kx[1],", kymin=",ky[1])


# In[ ]:


from numba import njit, prange

@njit(inline="always")
def annotate_directional_kernel(annot,D_kpq,mx,my,qx,qy,screening=0.1):
    """
    Core function to annotate directional energy transfer in a spectral space.
    
    Parameters:
    annot : list
        List to store annotations (arrows representing energy transfer).
    D_kpq : numpy array
        Array representing energy transfer in spectral space.
    mx, my, qx, qy : int
        Indices in the spectral space.
    screening : float, optional
        Threshold for displaying energy transfer (default is 0.1).
    """
    nky = int((D_kpq.shape[0]-1)/2)
    nkx = int((D_kpq.shape[1]-1)/2)
    px=-mx-qx
    py=-my-qy
    if (abs(px)>nkx or abs(py)>nky):
        pass  # S_k^pq is not defined.
    else:
        wj = D_kpq[my,mx,qy,qx] # D_{k<-q}^p
        amplitude = np.abs(wj)
        if amplitude > screening: # Screening for visibility
#             if wj < 0: # k(loss)->q(gain)
#                 x_end = kx[qx]   *np.sign(ky[qy]+0.5*ky[1]) # Plot for ky >= 0
#                 y_end = ky[qy]   *np.sign(ky[qy]+0.5*ky[1]) # Plot for ky >= 0
#                 x_start = kx[mx] *np.sign(ky[my]+0.5*ky[1]) # Plot for ky >= 0
#                 y_start = ky[my] *np.sign(ky[my]+0.5*ky[1]) # Plot for ky >= 0
#             else: # k(gain)<-q(loss)
#                 x_end = kx[mx]   *np.sign(ky[my]+0.5*ky[1]) # Plot for ky >= 0
#                 y_end = ky[my]   *np.sign(ky[my]+0.5*ky[1]) # Plot for ky >= 0
#                 x_start = kx[qx] *np.sign(ky[qy]+0.5*ky[1]) # Plot for ky >= 0
#                 y_start = ky[qy] *np.sign(ky[qy]+0.5*ky[1]) # Plot for ky >= 0
            if wj < 0: # k(loss)->q(gain)
                x_end = kx[qx]   *np.sign(kx[qx]+0.5*kx[1]) # Plot for kx >= 0
                y_end = ky[qy]   *np.sign(kx[qx]+0.5*kx[1]) # Plot for kx >= 0
                x_start = kx[mx] *np.sign(kx[mx]+0.5*kx[1]) # Plot for kx >= 0
                y_start = ky[my] *np.sign(kx[mx]+0.5*kx[1]) # Plot for kx >= 0
            else: # k(gain)<-q(loss)
                x_end = kx[mx]   *np.sign(kx[mx]+0.5*kx[1]) # Plot for kx >= 0
                y_end = ky[my]   *np.sign(kx[mx]+0.5*kx[1]) # Plot for kx >= 0
                x_start = kx[qx] *np.sign(kx[qx]+0.5*kx[1]) # Plot for kx >= 0
                y_start = ky[qy] *np.sign(kx[qx]+0.5*kx[1]) # Plot for kx >= 0
                
            if x_start**2+y_start**2 < x_end**2+y_end**2: # low-k -> high-k: normal cascade
                color = amplitude
            else: # low-k <- high-k: inverse cascade
                color = -amplitude
            annot.append([x_end,y_end,x_start,y_start,color])
    return annot

def convert_color(energy):
    """
    Coloring node by its energy
    
    Parameter
    ---------
    energy : Numpy array
        Energy of the node
    
    Returns
    -------
    fillcolor : str
        Color of the node in RBGA
    """
    if energy==0:
        fillcolor="#00000000"
    else:
        energy=(energy+1)/2
        if energy>1:
            energy=1
        elif energy<0:
            energy=0
        cm = plt.get_cmap("seismic",256)
        r,g,b,a=cm(min(int(255*energy),255))
        #r,g,b,a=int(255*r),int(255*g),int(255*b),int(255*a)
        r,g,b,a=int(255*r),int(255*g),int(255*b),int(255*a*(np.abs(energy-0.5)*2))
        fillcolor="#{:02x}{:02x}{:02x}{:02x}".format(r,g,b,a)
    return fillcolor

# def convert_color2(energy):
#     """
#     Coloring node by its energy
    
#     Parameter
#     ---------
#     energy : Numpy array
#         Energy of the node
    
#     Returns
#     -------
#     fillcolor : str
#         Color of the node in RBGA
#     """
#     if energy==0:
#         fillcolor="#00000000"
#     else:
#         if energy > 0:
#             r,g,b,a=int(255),int(0),int(0),int(255)
#         else:
#             r,g,b,a=int(0),int(0),int(255),int(255)
#         fillcolor="#{:02x}{:02x}{:02x}{:02x}".format(r,g,b,a)
#     return fillcolor

@njit
def annotate_directional_kpq(D_kpq,mx_in,my_in,qx_in,qy_in,screening=0.0):
    """
    Annotates directional energy transfer for a specific set of indices in spectral space.

    Parameters:
    D_kpq : numpy array
        Array representing energy transfer in spectral space.
    mx_in, my_in, qx_in, qy_in : int
        Indices in the spectral space for the calculation.
    screening : float, optional
        Threshold for displaying energy transfer (default is 0.0).
    
    Returns:
    numpy array : Annotations for plotting, including start and end points, and color information.
    """
    nky = int((D_kpq.shape[0]-1)/2)
    nkx = int((D_kpq.shape[1]-1)/2)
    px_in=-mx_in-qx_in
    py_in=-my_in-qy_in
    
    annot = []
    annotate_directional_kernel(annot,D_kpq,mx_in,my_in,qx_in,qy_in,screening)
    annotate_directional_kernel(annot,D_kpq,px_in,py_in,mx_in,my_in,screening)
    annotate_directional_kernel(annot,D_kpq,qx_in,qy_in,px_in,py_in,screening)
    return np.array(annot)

@njit
def annotate_directional_all(D_kpq,screening=0.1):
    """
    Annotates directional energy transfer for all possible indices in the spectral space.

    Parameters:
    D_kpq : numpy array
        Array representing energy transfer in spectral space.
    screening : float, optional
        Threshold for displaying energy transfer (default is 0.1).
    
    Returns:
    numpy array : Annotations for plotting, including start and end points, and color information.
    """
    nky = int((D_kpq.shape[0]-1)/2)
    nkx = int((D_kpq.shape[1]-1)/2)

    # add edges
    annot = []
    for my in range(-nky,nky+1):
        for mx in range(-nkx,nkx+1):
            for qy in range(-nky,nky+1):
                if (abs(-my-qy)<=nky):
                    for qx in range(-nkx,nkx+1):
                        if (abs(-my-qx)<=nkx):
                            annotate_directional_kernel(annot,D_kpq,mx,my,qx,qy,screening) # Inline expansion for Numba
    return np.array(annot)

@njit
def annotate_directional_aroundk(D_kpq,mx,my,screening=0.1):
    """
    Annotates directional energy transfer around a specific k point in the spectral space.

    Parameters:
    D_kpq : numpy array
        Array representing energy transfer in spectral space.
    mx, my : int
        Indices in the spectral space for the center point.
    screening : float, optional
        Threshold for displaying energy transfer (default is 0.1).
    
    Returns:
    numpy array : Annotations for plotting, including start and end points, and color information.
    """
    nky = int((D_kpq.shape[0]-1)/2)
    nkx = int((D_kpq.shape[1]-1)/2)

    # add edges
    annot = []
    for qy in range(-nky,nky+1):
        if (abs(-my-qy)<=nky):
            for qx in range(-nkx,nkx+1):
                if (abs(-my-qx)<=nkx):
                    annotate_directional_kernel(annot,D_kpq,mx,my,qx,qy,screening) # Inline expansion for Numba
    return np.array(annot)


# In[ ]:


annot = annotate_directional_all(D_kpq,screening=0.30)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
for i in range(annot.shape[0]):
    ax.annotate("",annot[i,0:2],annot[i,2:4],
                arrowprops = dict(width=annot[i,4]*8,
                                  headwidth=annot[i,4]*20,
                                  headlength=annot[i,4]*20,
                                  edgecolor=convert_color(annot[i,4]),
                                  facecolor=convert_color(annot[i,4])
                                 ))
ax.set_title(r"Directional transfer $D_{k \leftarrow q}^{p}$ (all)")
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
ax.set_xlim(0,2)
ax.set_ylim(-2,2)
ax.set_aspect("equal")
plt.show()


# In[ ]:


mx=5
my=0
qx=2
qy=-8
px=-mx-qx
py=-my-qy
print("#Check the detailed balance: D_kpq = - D_qpk")
wD_kpq=float(D_kpq[my,mx,qy,qx])
wD_pqk=float(D_kpq[py,px,my,mx])
wD_qkp=float(D_kpq[qy,qx,py,px])
print("D_kpq=",wD_kpq,"#(kx,ky)=({:},{:})".format(kx[mx],ky[my]),", D_qpk=",float(D_kpq[qy,qx,my,mx]))
print("D_pqk=",wD_pqk,"#(px,py)=({:},{:})".format(kx[px],ky[py]),", D_kqp=",float(D_kpq[my,mx,py,px]))
print("D_qkp=",wD_qkp,"#(qx,qy)=({:},{:})".format(kx[qx],ky[qy]),", D_pkq=",float(D_kpq[py,px,qy,qx]))

annot = annotate_directional_kpq(D_kpq,mx_in=mx,my_in=my,qx_in=qx,qy_in=qy,screening=0.0)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
for i in range(annot.shape[0]):
    ax.annotate("",annot[i,0:2],annot[i,2:4],
                arrowprops = dict(width=annot[i,4]*8,
                                  headwidth=annot[i,4]*20,
                                  headlength=annot[i,4]*20,
                                  edgecolor=convert_color(annot[i,4]),
                                  facecolor=convert_color(annot[i,4])
                                 ))
ax.set_title(r"Directional transfer $D_{k \leftarrow q}^{p}$ (Single k,p,q)")
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
ax.set_xlim(0,2)
ax.set_ylim(-2,2)
ax.set_aspect("equal")
plt.show()


# In[ ]:


mx=5
my=0
annot = annotate_directional_aroundk(D_kpq,mx,my,screening=0.05)
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
for i in range(annot.shape[0]):
    ax.annotate("",annot[i,0:2],annot[i,2:4],
                arrowprops = dict(width=annot[i,4]*8,
                                  headwidth=annot[i,4]*20,
                                  headlength=annot[i,4]*20,
                                  edgecolor=convert_color(annot[i,4]),
                                  facecolor=convert_color(annot[i,4])
                                 ))
# ax.set_title(r"Directional transfer $D_{k \leftarrow q}^{p}$")
ax.set_title(r"$D_{k \leftarrow q}^{p}$ [for $(k_x,k_y)=$"+"$({:},{:})]$".format(kx[mx],ky[my]))
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
ax.set_xlim(0,2)
ax.set_ylim(-2,2)
ax.set_aspect("equal")
plt.show()

diag_mx=mx
diag_my=my
fig = plt.figure()
ax = fig.add_subplot(111)
xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$q_x$")
ax.set_ylabel(r"$q_y$")
ax.set_aspect("equal")
ax.set_title(r"$D_{k \leftarrow q}^{p}$ [for $(k_x,k_y)=$"+"$({:.1f},{:.1f})]$".format(kx[diag_mx],ky[diag_my]))
kx_shift = np.fft.fftshift(kx)
ky_shift = np.fft.fftshift(ky)
D_kpq_shift = np.fft.fftshift(D_kpq[diag_my,diag_mx,:,:])
vmax = abs(D_kpq_shift).max()
quad = ax.pcolormesh(kx_shift,ky_shift,D_kpq_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
theta=np.linspace(-np.pi,np.pi,100)
kabs=np.sqrt(kx[diag_mx]**2+ky[diag_my]**2)
ax.plot(kabs*np.cos(theta),kabs*np.sin(theta),linestyle="--",color="k",linewidth=1)
ax.scatter(kx[diag_mx],ky[diag_my],marker="x",color="k")
fig.colorbar(quad)
plt.show() 


# In[ ]:


mx=5
my=10
annot = annotate_directional_aroundk(D_kpq,mx,my,screening=0.05)
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
for i in range(annot.shape[0]):
    ax.annotate("",annot[i,0:2],annot[i,2:4],
                arrowprops = dict(width=annot[i,4]*8,
                                  headwidth=annot[i,4]*20,
                                  headlength=annot[i,4]*20,
                                  edgecolor=convert_color(annot[i,4]),
                                  facecolor=convert_color(annot[i,4])
                                 ))
# ax.set_title(r"Directional transfer $D_{k \leftarrow q}^{p}$")
ax.set_title(r"$D_{k \leftarrow q}^{p}$ [for $(k_x,k_y)=$"+"$({:},{:})]$".format(kx[mx],ky[my]))
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
ax.set_xlim(0,2)
ax.set_ylim(-2,2)
ax.set_aspect("equal")
plt.show()

diag_mx=mx
diag_my=my
fig = plt.figure()
ax = fig.add_subplot(111)
xylim=2
ax.set_xlim(-xylim,xylim)
ax.set_ylim(-xylim,xylim)
ax.set_xlabel(r"$q_x$")
ax.set_ylabel(r"$q_y$")
ax.set_aspect("equal")
ax.set_title(r"$D_{k \leftarrow q}^{p}$ [for $(k_x,k_y)=$"+"$({:.1f},{:.1f})]$".format(kx[diag_mx],ky[diag_my]))
kx_shift = np.fft.fftshift(kx)
ky_shift = np.fft.fftshift(ky)
D_kpq_shift = np.fft.fftshift(D_kpq[diag_my,diag_mx,:,:])
vmax = abs(D_kpq_shift).max()
quad = ax.pcolormesh(kx_shift,ky_shift,D_kpq_shift,
                     cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
theta=np.linspace(-np.pi,np.pi,100)
kabs=np.sqrt(kx[diag_mx]**2+ky[diag_my]**2)
ax.plot(kabs*np.cos(theta),kabs*np.sin(theta),linestyle="--",color="k",linewidth=1)
ax.scatter(kx[diag_mx],ky[diag_my],marker="x",color="k")
fig.colorbar(quad)
plt.show() 


# In[ ]:




