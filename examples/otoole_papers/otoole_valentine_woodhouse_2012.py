import pyprop8 as pp
from pyprop8.utils import rtf2xyz,make_moment_tensor,stf_trapezoidal,stf_cosine,latlon2xy,clp_filter
import numpy as np
import matplotlib.pyplot as plt
import torch as tp
from tqdm import tqdm

'''This example aims to reproduce the first couple of figures presented in
O'Toole, Valentine & Woodhouse (2012, doi: 10.1111/j.1365-246X.2012.05608.x).

As various details are not fully and unambiguously defined in the paper, there
may be some minor differences between the figures there and those output by this
code.
'''
figpath = './figures'
# Table 1:
model = pp.LayeredStructureModel([[ 0.10, 3.20, 2.00, 2.10],
                                           [ 1.90, 5.15, 2.85, 2.50],
                                           [ 3.00, 5.50, 3.20, 2.60],
                                           [13.00, 6.00, 3.46, 2.70],
                                           [14.00, 6.70, 3.87, 2.80],
                                           [np.inf,7.70, 4.30, 3.30]]
                                           )
print(model)

stations = pp.RegularlyDistributedReceivers(39.6,39.6,5,90-118.2,90-118.2,5,degrees=True).asListOfReceivers()

# Table 2, 'Iteration 0' column

Mrtf = tp.tensor([[ 0.3406, 0.0005, 0.1610],
                [ 0.0005, 0.7798, 0.1430],
                [ 0.1610, 0.1430, 0.6349]]).requires_grad_()

event = pp.PointSource(tp.tensor(0),tp.tensor(0),tp.tensor(35),rtf2xyz(Mrtf),tp.tensor([[0.],[0.],[0.]]),0)
drv = pp.DerivativeSwitches(moment_tensor=True,z=True,x=True,y=True,time=True)

# Appears that the paper just uses the cosine low-pass filter
# and not any further source time-function

stf = lambda w: clp_filter(w,0.05*2*np.pi,0.2*2*np.pi)
tt,seis,deriv = pp.compute_seismograms(model,event,stations,81,0.5,source_time_function = stf,derivatives=drv,pad_frac=0.5)
seis = seis[0]
deriv = deriv[0]

pt_derivs = tp.zeros_like(deriv) #(6,3,nt)
deriv = deriv.detach().numpy()
for i in range(seis.shape[0]):
    for j in tqdm(range(seis.shape[1])):
        g0 = tp.zeros_like(seis)
        g0[i,j] = 1
        seis.backward(g0,retain_graph=True)
        grad = Mrtf.grad
        #   Just diagonal for now
        for k in range(3):
            pt_derivs[k,i,j] = grad[k,k]
        inds = [[0,1],[0,2],[1,2]]
      
        k=3
        for ind in inds:
            pt_derivs[k,i,j] = grad[ind[0],ind[1]]
            pt_derivs[k,i,j] += grad[ind[1],ind[0]]
            k+=1

        Mrtf.grad = tp.zeros_like(Mrtf)

pt_derivs = pt_derivs.detach().numpy()
seis = seis.detach().numpy()
nez = [1,0,2] #Reorder seismogram components to match O'Toole's figure
# Native ordering of moment tensor components is as follows:
#    Mxx, Myy, Mzz, Mxy, Mxz, Myz
# Geometrical conversions:
#    dx -> dphi
#    dy -> -dtheta
#    dz -> dr
# Hence native ordering is equivalent to:
#    Mpp, Mtt, Mrr, -Mtp, Mrp -Mrt
dcomp = [2,1,0,5,4,3]
dcompsign = [1,1,1,-1,1,-1]
complabel=[r"$M_{rr}$ or $M_{zz}$",
           r"$M_{\theta\theta}$ or $M_{yy}$",
           r"$M_{\phi\phi}$ or $M_{xx}$",
           r"$M_{r\theta}$ or $-M_{yz}$",
           r"$M_{r\phi}$ or $M_{xz}$",
           r"$M_{\theta\phi}$ or $-M_{xy}$"]

amax = 1.025*abs(deriv[drv.i_mt:drv.i_mt+6,:,:]).max()
fig = plt.figure()
for idrv in range(6):
    for icomp in range(3):
        ax = fig.add_subplot(6,3,(3*idrv)+icomp+1)
        if idrv==0:
            if icomp == 0:
                ax.set_title("North")
            elif icomp == 1:
                ax.set_title("East")
            elif icomp == 2:
                ax.set_title("Vertical")
        ax.plot(tt,np.zeros_like(tt),'k:')
        proxyp = ax.plot(tt,dcompsign[idrv]*deriv[drv.i_mt+dcomp[idrv],nez[icomp],:])
        proxy = ax.plot(tt,pt_derivs[drv.i_mt+idrv,nez[icomp],:],'r--')
        ax.set_ylim(-amax,amax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        if icomp==0: ax.text(0,.7,complabel[idrv],transform=ax.transAxes)
proxy[0].set_label('Auto-Diff')
proxyp[0].set_label('pyprop8')
ax.legend()
plt.tight_layout()
plt.savefig(f'{figpath}/OVW12_fig1.png',dpi=256)
plt.show()


fig = plt.figure()
deriv[drv.i_z,:,:]*=10
deriv[drv.i_x,:,:]*=10
deriv[drv.i_y,:,:]*=10
deriv[drv.i_time,:,:]*=2



amax = abs(deriv[drv.i_mt+6:,:,:]).max()
amax = 1.025*max([amax,abs(seis).max()])
dcomp = [drv.i_z,drv.i_y,drv.i_x,drv.i_time]
complabel = ["Depth","Latitude","Longitude","Time"]
for idrv in range(4):
    for icomp in range(3):
        ax = fig.add_subplot(5,3,(3*idrv)+icomp+1)
        if idrv==0:
            if icomp == 0:
                ax.set_title("North")
            elif icomp == 1:
                ax.set_title("East")
            elif icomp == 2:
                ax.set_title("Vertical")
        ax.plot(tt,np.zeros_like(tt),'k:')
        ax.plot(tt,deriv[dcomp[idrv],nez[icomp],:])
        ax.set_ylim(-amax,amax)
        ax.axis('off')
        if icomp==0: ax.text(0,.7,complabel[idrv],transform=ax.transAxes)
for icomp in range(3):
    ax = fig.add_subplot(5,3,13+icomp)
    ax.plot(tt,np.zeros_like(tt),'k:')
    ax.plot(tt,seis[nez[icomp],:])
    ax.set_ylim(-amax,amax)
    ax.axis('off')
    if icomp==0: ax.text(0,.7,"HRGPS synthetic",transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f'{figpath}/OVW12_fig2.png',dpi=256)
plt.show()
