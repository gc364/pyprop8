import pyprop8 as pp
from pyprop8.utils import stf_trapezoidal, make_moment_tensor, rtf2xyz
import torch as np
import matplotlib.pyplot as plt

np.set_default_dtype(np.float64)
def tests():
    print("Running tests. Using `pyprop8` from: %s" % pp.__file__)
    print("")
    print(" 1. Creating objects")
    
    #################################################
    #       Structure setup and gradient init       #
    #################################################
    model = pp.LayeredStructureModel(
        [
            (3.0, 1.8, 0.0, 1.02),
            (2.0, 4.5, 2.4, 2.57),
            (5.0, 5.8, 3.3, 2.63),
            (20.0, 6.5, 3.65, 2.85),
            (np.inf, 8.0, 4.56, 3.34),
        ]
    )
    model.sigma.requires_grad_()
    model.mu.requires_grad_()
    model.rho.requires_grad_()
    model.dz.requires_grad_()
    
    ######################################################
    ################Source Param Grad test################
    ######################################################
    ######################################################
    strike = np.deg2rad(np.tensor(340.)).requires_grad_()
    dip = np.deg2rad(np.tensor(70.)).requires_grad_()
    rake = np.deg2rad(np.tensor(20.)).requires_grad_()
    m0=np.tensor(2.4e8).requires_grad_()
    eta =np.tensor(0.).requires_grad_()
    xtr =np.tensor(0.).requires_grad_()

    Mrtp = make_moment_tensor(strike,dip,rake,m0,eta,xtr)
    Mxyz = rtf2xyz(Mrtp)
    Mxyz.retain_grad()
   
    x = np.tensor(0.)
    y =  np.tensor(0.)
    d = np.tensor(20.).requires_grad_()
    F = np.zeros([3, 1]).requires_grad_() 
    t = np.tensor(0.)


    source = pp.PointSource(
        x,
        y,
        d,
        Mxyz,
        F,
        t,
    )
    #################################################################
    #################################################################
    ################################################################
    stations = pp.RegularlyDistributedReceivers(
        30, 100, 7, 0, 360, 10, depth=3
    ).asListOfReceivers()

    derivs = pp.DerivativeSwitches(x=True, y=True, z=True)

    source_time_function = lambda w: stf_trapezoidal(w, 3, 6)

    print(" 2. Computing seismograms...")

    nt = 60  # 257
    dt = 1
    alpha = 0.023
    pad_frac = 1
    kwargs = {"kmin": 0, "kmax": 2.04, "nk": 1200} #nk=1200 is the default, decrease for H-matrix estimation
    tt, seis0,drv = pp.compute_seismograms(
        model,
        source,
        stations,
        nt,
        dt,
        alpha,
        pad_frac=pad_frac,
        derivatives=derivs,
        source_time_function=source_time_function,
        xyz=True,
        **kwargs
    )

    epsilon = 1e-4
    print(" 3. Comparing with finite-difference derivatives.")
    print(
        "    Using finite-difference perturbation eps = %.2f metres" % (epsilon * 1000)
    )
    print("    a. Perturbing source in x.")

    source_x = source.clone()
    source_x.x = source_x.x+epsilon

    tt, seis_x = pp.compute_seismograms(
        model,
        source_x,
        stations,
        nt,
        dt,
        alpha,
        pad_frac=pad_frac,
        derivatives=None,
        source_time_function=source_time_function,
        xyz=True,
        **kwargs
    )

    fd = (seis_x - seis0) / epsilon  # finite difference estimate
    max_x = (
        abs(drv[:, derivs.i_x, :, :]).max(-1).values.reshape(stations.nstations, 3, 1)
    )  # Maximum absolute value of trace
    perc_err_x = 100 * (
        abs(drv[:, derivs.i_x, :, :] - fd) / max_x
    )  # For each trace, calculate finite difference 'error' as percentage of trace amplitude
    print(
        "       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"
        % perc_err_x.max()
    )

    print("    b. Perturbing source in y.")
    source_y = source.clone()
    source_y.y =source_y.y + epsilon

    tt, seis_y = pp.compute_seismograms(
        model,
        source_y,
        stations,
        nt,
        dt,
        alpha,
        pad_frac=pad_frac,
        derivatives=None,
        source_time_function=source_time_function,
        xyz=True,
        **kwargs
    )

    fd = (seis_y - seis0) / epsilon  # finite difference estimate
    max_y = (
        abs(drv[:, derivs.i_y, :, :]).max(-1).values.reshape(stations.nstations, 3, 1)
    )  # Maximum absolute value of trace
    perc_err_y = 100 * (
        abs(drv[:, derivs.i_y, :, :] - fd) / max_y
    )  # For each trace, calculate finite difference 'error' as percentage of trace amplitude
    print(
        "       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"
        % perc_err_y.max()
    )

    print("    c. Perturbing source in z.")
    source_z = source.clone()
    source_z.dep =source_z.dep -epsilon  # coordinate system is z-up so a positive epsilon in z is a *reduction* in source depth

    tt, seis_z = pp.compute_seismograms(
        model,
        source_z,
        stations,
        nt,
        dt,
        alpha,
        pad_frac=pad_frac,
        derivatives=None,
        source_time_function=source_time_function,
        xyz=True,
        **kwargs
    )

    fd = (seis_z - seis0) / epsilon  # finite difference estimate
    max_z = (
        abs(drv[:, derivs.i_z, :, :]).max(-1).values.reshape(stations.nstations, 3, 1)
    )  # Maximum absolute value of trace
    perc_err_z = 100 * (
        abs(drv[:, derivs.i_z, :, :] - fd) / max_z
    )  # For each trace, calculate finite difference 'error' as percentage of trace amplitude
    print(
        "       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"
        % perc_err_z.max()
    )

    print(" 4. Computing static displacement field")
    stat0, drv = pp.compute_static(
        model,
        source,
        stations,
        derivatives=derivs,
    )

    print(" 5. Comparing with finite-difference derivatives")
    print("    a. Perturbing source in x.")

    stat_x = pp.compute_static(
        model,
        source_x,
        stations,
    )

    fd = (stat_x - stat0) / epsilon
    max_x = abs(drv[:, derivs.i_x, :]).max(0).values.reshape(1, 3)
    perc_err_x = 100 * abs(drv[:, derivs.i_x, :] - fd) / max_x
    print(
        "       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"
        % (perc_err_x.max())
    )
    print("    b. Perturbing source in y.")

    stat_y = pp.compute_static(
        model,
        source_y,
        stations,
    )
    fd = (stat_y - stat0) / epsilon

    max_y = abs(drv[:, derivs.i_y, :]).max(0).values.reshape(1, 3)
    perc_err_y = 100 * abs(drv[:, derivs.i_y, :] - fd) / max_y
    print(
        "       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"
        % (perc_err_y.max())
    )
    print("    c. Perturbing source in z.")

    stat_z = pp.compute_static(
        model,
        source_z,
        stations,
    )
    fd = (stat_z - stat0) / epsilon
    max_z = abs(drv[:, derivs.i_z, :]).max(0).values.reshape(1, 3)
    perc_err_z = 100 * abs(drv[:, derivs.i_z, :] - fd) / max_z
    print(
        "       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"
        % (perc_err_z.max())
    )
 
    ##################################################
    #####One step optimisation to test autodiff#######
    ##################################################
    np.autograd.set_detect_anomaly(True)
    print('INPUT PARAMETERS')
    params = [strike,dip,rake,m0,eta,xtr,d,F]
    labels = ['strike','dip','rake','M0','eta','xtr','d','F']
    print(params)
    print(Mxyz)

    print('SOURCE SENSITIVITY GRADIENTS')
    seis0.backward(np.ones_like(seis0),retain_graph=True)
    print(Mxyz.grad)
    for l,p in zip(labels,params):
        print(f'{l}: {p.grad}')
    print('STRUCTURE SENSITIVITY GRADIENTS')
    print(fr'{r'$\delta$z'}: {model.dz.grad}')
    print(fr'{r'$\mu$'}: {model.mu.grad}')
    print(fr'{r'$\sigma$'}: {model.sigma.grad}')
    print(fr'{r'$\rho$'}: {model.rho.grad}')


    for p in params:
        p.grad = np.ones_like(p)
    
    print('SOURCE OPTIMISATION GRADIENTS')
    loss_fn = np.nn.L1Loss()
    optim = np.optim.Adam(params,1)
    l = loss_fn(np.zeros_like(seis0,dtype=np.complex128),seis0)
    l.backward()
    optim.step()
    
    print(Mxyz.grad)
    for l,p in zip(labels,params):
        print(f'{l}: {p.grad}')
    

    fig,ax = plt.subplots()
    ax.plot(tt.detach().numpy(),seis0[35,0].detach().numpy(),label='x')
    ax.plot(tt.detach().numpy(),seis0[35,1].detach().numpy(),label='y')
    ax.plot(tt.detach().numpy(),seis0[35,2].detach().numpy(),label='z')
    plt.legend()
    plt.show()
    
    


if __name__ == "__main__":
    tests()
