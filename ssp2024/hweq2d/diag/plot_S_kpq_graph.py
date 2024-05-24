#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_mfdataset("./data_netcdf/S_kpq_0000-0099.nc")
S_kpq = np.array(ds.S_kpq)
# print(S_kpq)
ky=np.array(ds.ky)
kx=np.array(ds.kx)
nkx=int((len(kx)-1)/2)
nky=int((len(ky)-1)/2)
#S_kpq_max = np.abs(S_kpq).max()
S_kpq_max = np.max([abs(S_kpq.max()),abs(S_kpq.min())])
S_kpq = S_kpq / S_kpq_max
print("S_kpq_max(abs):", S_kpq_max)
print("S_kpq_max(normalized):",S_kpq.max(),", S_kpq_min(normalized):",S_kpq.min())


# In[ ]:


import pygraphviz as pgv
from IPython.display import display, SVG

def mxmy2idx(mx,my):
    return mx+nkx+(2*nkx+1)*(my+nky)

def idx2mxmy(idx):
    mx = np.mod(idx, 2*nkx+1)-nkx
    my = int(idx/(2*nkx+1))-nky
    return mx,my

# Plot symmetric triad transfer S_k^pq
def triadgraph_symmetric_kernel(G,trans,mx,my,qx,qy,screening,pwidth,nodename):
    """
    Draw network of symmetric triad transfer
    
    Parameters
    ----------
    G : AGraph of pygraphviz
        G = pygraphviz.AGraph(directed=True,strict=False)
    trans : Numpy array
        The symmetric triad transfer function S_k^pq
        Its shape is (n,n,n) where n is the number of modes.
        Its amplitude should be normalized to draw a graph.
    k : int
        index of S_k^pq
    p : int
        index of S_k^pq
    q : int
        index of S_k^pq
    screening : float
        For visibility, draw edges only for |S_k^pq| > screening.
    pwidth : float
        Penwidth for drawing edges, penwidth=pwidth*|S_k^pq|.
    nodename : list
        List of node name, len(nodename) = n where n is the number of modes.
    
    Returns
    -------
    G : AGraph of pygraphviz
        Nodes and edges are added, representing S_k^pq + S_p^qk + S_q^kp = 0
    """
    nky = int((trans.shape[0]-1)/2)
    nkx = int((trans.shape[1]-1)/2)
    px=-mx-qx
    py=-my-qy
    if (abs(px)>nkx or abs(py)>nky):
        pass  # S_k^pq is not defined.
    elif ((mx,my)==(px,py) or (px,py)==(qx,qy) or (qx,qy)==(mx,my)):
        pass  # S_k^pq = 0
    else:
        id_m = mxmy2idx(mx,my)
        id_p = mxmy2idx(px,py)
        id_q = mxmy2idx(qx,qy)
        min_id = min(id_m,id_p,id_q)
        max_id = max(id_m,id_p,id_q)
        mid_id = id_m+id_p+id_q-min_id-max_id
        min_x,min_y = idx2mxmy(min_id)
        mid_x,mid_y = idx2mxmy(mid_id)
        max_x,max_y = idx2mxmy(max_id)
        kpq_junction = "{},{},{}".format(nodename[min_y][min_x],nodename[mid_y][mid_x],nodename[max_y][max_x])
        wj = np.array([trans[my,mx,qy,qx], trans[py,px,my,mx], trans[qy,qx,py,px]]) # S_kpq, S_pqk, S_qkp
        if np.abs(2*wj[:]).max() > screening:
            G.add_node(kpq_junction,shape="circle",fixedsize=True,width=0.05,height=0.05,
                       style="filled",fillcolor="black",label="")
        # The detailed balance: S_k^pq + S_p^qk + S_q^kp = 0
        # Net transfer via triad interaction among (k,p,q) is 2*S_k^pq=S_k^pq+S_k^qp
        if np.abs(2*wj[0]) > screening: # Screening for visibility
            if wj[0] < 0:
                G.add_edge(nodename[my][mx],kpq_junction,penwidth=pwidth*abs(2*wj[0]))
            else:
                G.add_edge(kpq_junction,nodename[my][mx],penwidth=pwidth*abs(2*wj[0]))
        if np.abs(2*wj[1]) > screening: # Screening for visibility
            if wj[1] < 0:
                G.add_edge(nodename[py][px],kpq_junction,penwidth=pwidth*abs(2*wj[1]))
            else:
                G.add_edge(kpq_junction,nodename[py][px],penwidth=pwidth*abs(2*wj[1]))
        if np.abs(2*wj[2]) > screening: # Screening for visibility
            if wj[2] < 0:
                G.add_edge(nodename[qy][qx],kpq_junction,penwidth=pwidth*abs(2*wj[2]))
            else:
                G.add_edge(kpq_junction,nodename[qy][qx],penwidth=pwidth*abs(2*wj[2]))

    return G



def triadgraph_symmetric_all(trans,kxmin=1.0,kymin=1.0,output=None,title=None,screening=0.1,pwidth=5.0,nodename=None,energy=None):
    """
    Draw network of symmetric triad transfer
    
    Parameters
    ----------
    trans : Numpy array
        The symmetric triad transfer function S_k^pq
        Its shape is (n,n,n) where n is the number of modes.
        Its amplitude should be normalized to draw a graph.
    output : str
        If output == None:
            show a network graph on display.
        else:
            save a png or dot file as path=output.
    title : str, optional
        Title of graph
    screening : float, optional
        For visibility, draw edges only for |S_k^pq| > screening.
        Default: screening=0.1
    pwidth : float, optional
        Penwidth for drawing edges, penwidth=pwidth*|S_k^pq|.
        Default: pwidth=5.0
    nodename : list, optional
        List of node name, len(nodename) = n where n is the number of modes.
    energy : Numpy array, optional
        Energy of the modes
        Its shape is (n) where n is the number of modes.
        Its amplitude should be normalized to draw a graph.
    """
    nky = int((trans.shape[0]-1)/2)
    nkx = int((trans.shape[1]-1)/2)
    if title is None:
        G = pgv.AGraph(directed=True,strict=True)
    else:
        G = pgv.AGraph(directed=True,strict=True,label=title)
    if nodename is None:
        nodename = []
        for my in np.roll(np.arange(-nky,nky+1),nky+1):
            wlist = []
            for mx in np.roll(np.arange(-nkx,nkx+1),nkx+1):
                wlist.append((mx*kxmin,my*kymin))
            nodename.append(wlist)
    G.node_attr["shape"]="ellipse" # Default node shape
    G.node_attr["fixedsize"]=True
    G.node_attr["width"]=1.0
    G.node_attr["height"]=0.5

#     # add nodes (Radial layout, color by energy)
#     if energy is None:
#         energy = np.zeros(n)
#     G.add_node(nodename[0],color='red',shape="diamond",pos="0,0",pin=True,\
#                style="filled",fillcolor=convert_energy2color(energy[0]))
#     for k in range(1,n):
#         theta = 2.0*np.pi*(k-1)/(n-1)
#         G.add_node(nodename[k],color='red',shape="diamond",pos="{},{}".format(-2*np.sin(theta),2*np.cos(theta)),pin=True,\
#                    style="filled",fillcolor=convert_energy2color(energy[k]))

    # add edges
    for my in range(-30,30+1):
        for mx in range(-30,30+1):
            for qy in range(-30,30+1):
                if (abs(-my-qy)<=nky):
                    for qx in range(-30,30+1):
                        if (abs(-my-qx)<=nkx):
                            triadgraph_symmetric_kernel(G,trans,mx,my,qx,qy,screening,pwidth,nodename)

    # draw network
    if output is None:
        img = G.draw(prog="fdp", format="svg")#prog=neato|dot|twopi|circo|fdp|nop.
        display(SVG(img))
    elif output[-3:]=="png":
        G.draw(path=output,prog="fdp",format="png")#prog=neato|dot|twopi|circo|fdp|nop.  
    elif output[-3:]=="dot":
        G.draw(path=output,prog="fdp",format="dot")#prog=neato|dot|twopi|circo|fdp|nop.  

    return



def triadgraph_symmetric_kpq(trans,mx_in,my_in,qx_in,qy_in,kxmin=1.0,kymin=1.0,output=None,title=None,screening=0.1,pwidth=5.0,nodename=None,energy=None):
    """
    Draw network of symmetric triad transfer
    
    Parameters
    ----------
    trans : Numpy array
        The symmetric triad transfer function S_k^pq
        Its shape is (n,n,n) where n is the number of modes.
        Its amplitude should be normalized to draw a graph.
    k_in : int
        index of S_k^pq
    p_in : int
        index of S_k^pq
    q_in : int
        index of S_k^pq
    output : str
        If output == None:
            show a network graph on display.
        else:
            save a png or dot file as path=output.
    title : str, optional
        Title of graph
    screening : float, optional
        For visibility, draw edges only for |S_k^pq| > screening.
        Default: screening=0.1
    pwidth : float, optional
        Penwidth for drawing edges, penwidth=pwidth*|S_k^pq|.
        Default: pwidth=5.0
    nodename : list, optional
        List of node name, len(nodename) = n where n is the number of modes.
    energy : Numpy array, optional
        Energy of the modes
        Its shape is (n) where n is the number of modes.
        Its amplitude should be normalized to draw a graph.
    """
    nky = int((trans.shape[0]-1)/2)
    nkx = int((trans.shape[1]-1)/2)
    if title is None:
        G = pgv.AGraph(directed=True,strict=False)
    else:
        G = pgv.AGraph(directed=True,strict=False,label=title)
    if nodename is None:
        nodename = []
        for my in np.roll(np.arange(-nky,nky+1),nky+1):
            wlist = []
            for mx in np.roll(np.arange(-nkx,nkx+1),nkx+1):
                wlist.append((mx*kxmin,my*kymin))
            nodename.append(wlist)
    G.node_attr["shape"]="ellipse" # Default node shape
    G.node_attr["fixedsize"]=True
    G.node_attr["width"]=1.0
    G.node_attr["height"]=0.5
    
#     # add nodes (Radial layout, color by energy)
#     if energy is None:
#         energy = np.zeros(n)
#     G.add_node(nodename[0],color='red',shape="diamond",pos="0,0",pin=True,\
#                style="filled",fillcolor=convert_energy2color(energy[0]))
#     for k in range(1,n):
#         theta = 2.0*np.pi*(k-1)/(n-1)
#         G.add_node(nodename[k],color='red',shape="diamond",pos="{},{}".format(-2*np.sin(theta),2*np.cos(theta)),pin=True,\
#                    style="filled",fillcolor=convert_energy2color(energy[k]))

    # add edges
    triadgraph_symmetric_kernel(G,trans,mx_in,my_in,qx_in,qy_in,screening,pwidth,nodename)

    # draw network
    if output is None:
        img = G.draw(prog="fdp", format="svg")#prog=neato|dot|twopi|circo|fdp|nop.
        display(SVG(img))
    elif output[-3:]=="png":
        G.draw(path=output,prog="fdp",format="png")#prog=neato|dot|twopi|circo|fdp|nop.  
    elif output[-3:]=="dot":
        G.draw(path=output,prog="fdp",format="dot")#prog=neato|dot|twopi|circo|fdp|nop.  

    return


# In[ ]:


triadgraph_symmetric_all(S_kpq,kxmin=kx[1],kymin=ky[1],title="S_kpq",screening=0.5)


# In[ ]:


# G = pgv.AGraph(directed=True,strict=False)
# G.add_node?
mx=4
my=1
qx=0
qy=5
px=-mx-qx
py=-my-qy
print("#Check the detailed balance")
wS_kpq=float(S_kpq[my,mx,qy,qx])
wS_pqk=float(S_kpq[py,px,my,mx])
wS_qkp=float(S_kpq[qy,qx,py,px])
print("S_kpq=",wS_kpq,"#(kx,ky)=({:},{:})".format(kx[mx],ky[my]))
print("S_pqk=",wS_pqk,"#(px,py)=({:},{:})".format(kx[px],ky[py]))
print("S_qkp=",wS_qkp,"#(qx,qy)=({:},{:})".format(kx[qx],ky[qy]))
print("total=",wS_kpq+wS_pqk+wS_qkp)

triadgraph_symmetric_kpq(S_kpq,mx_in=mx,my_in=my,qx_in=qx,qy_in=qy,title="S_kpq",screening=0.0)


# In[ ]:




