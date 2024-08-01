import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
ifft = np.fft.ifft
fft = np.fft.fft
from ipywidgets import IntProgress
from IPython.display import display
import time

g=9.8

def bracket(f):
    """
    Returns a function that computes the bracket of a given function f.

    Parameters:
    f (function): The function to compute the bracket of.

    Returns:
    function: A function that computes the bracket of f.
    """
    mean = quad(f,0,1)[0]
    brace = lambda y: f(y)-mean
    brack_nzm = lambda y: quad(brace,0,y)[0]
    mean_bracket = quad(brack_nzm,0,1)[0]
    def brack(y):
        return quad(brace,0,y)[0] - mean_bracket
    return brack

def spectral_representation(x0,uhat,xi):
    """
    Returns a truncated Fourier series representation of a function.

    Parameters:
    x0 (float): The left endpoint of the domain of the function.
    uhat (numpy.ndarray): The Fourier coefficients of the function.
    xi (numpy.ndarray): The vector of wavenumbers.

    Returns:
    u_fun: A vectorized function that represents the Fourier series.
    """
    u_fun = lambda y : np.real(np.sum(uhat*np.exp(1j*xi*(y+x0))))/len(uhat)
    u_fun = np.vectorize(u_fun)
    return u_fun

def fine_resolution(f, n, x, xi):
    """
    Interpolates a periodic function `f` onto a finer grid of `n` points using a Fourier series.

    Parameters:
    -----------
    f : function
        The function to be interpolated.
    n : int
        The number of points in the finer grid.
    x : array-like
        The original grid of `f`.
    xi : array-like
        The Fourier modes.

    Returns:
    --------
    x_fine : array-like
        The finer grid of `n` points.
    f_spectral : function
        The Fourier interpolation `f` on the finer grid.
    """
def fine_resolution(f,n,x,xi):
    fhat = fft(f)
    f_spectral = spectral_representation(x[0],fhat,xi)
    x_fine = np.linspace(x[0],x[-1],n)
    return x_fine, f_spectral(x_fine)

def rk3(u,xi,rhs,dt,du,params=None):
    """
    Third-order Runge-Kutta time-stepping method for solving ODEs.

    Parameters:
    u (numpy.ndarray): The current solution.
    xi (numpy.ndarray): The spatial grid.
    rhs (function): The right-hand side of the ODE system.
    dt (float): The time step size.
    du (float): The spatial step size.
    params (dict, optional): Additional parameters to pass to the RHS function.

    Returns:
    numpy.ndarray: The updated solution at the next time step.
    """
    y2 = u + dt*rhs(u,du,xi,**params)
    y3 = 0.75*u + 0.25*(y2 + dt*rhs(y2,du,xi,**params))
    u_new = 1./3 * u + 2./3 * (y3 + dt*rhs(y3,du,xi,**params))
    return u_new

def rkm(u,xi,rhs,dt,du,fy,method,params=None):
    A = method.A
    b = method.b
    for i in range(len(b)):
        y = u.copy()
        for j in range(i):
            y += dt*A[i,j]*fy[j,:,:]
        fy[i,:,:] = rhs(y,du,xi,**params)
    #u_new = u + dt*sum([b[i]*fy[i,:,:] for i in range(len(b))])
    u_new = u + dt*np.sum(b[:,np.newaxis,np.newaxis]*fy, axis=0) # faster
    return u_new


def xxt_rhs(u, du, xi, **params):
    """
    Solves the BBM-like equation for a given set of parameters.

    Args:
        u (ndarray): Array of shape (2, N) containing the values of eta and q.
        du (ndarray): Array of shape (2, N) containing the derivatives of eta and q.
        xi (ndarray): Array of shape (N,) containing the values of xi.
        **params: Dictionary containing the values of the parameters H1, H2, H3, H4, alpha1, alpha2, alpha3, alpha4,
                  alpha5, alpha6, alpha7, alpha8, alpha9, alpha10, alpha11, C3, C11, delta, and order4.

    Returns:
        ndarray: Array of shape (2, N) containing the derivatives of eta and q.
    """
    H1, H2, H3, H4 = params['H1'], params['H2'], params['H3'], params['H4']
    alpha1, alpha2, alpha3 = params['alpha1'], params['alpha2'], params['alpha3']
    alpha4, alpha5, alpha6, alpha7, alpha8 = params['alpha4'], params['alpha5'], params['alpha6'], params['alpha7'], params['alpha8']
    alpha9, alpha10, alpha11 = params['alpha9'], params['alpha10'], params['alpha11']
    C3 = params['C3']
    delta = params['delta']
    mu = params['mu']

    eta = u[0,:]
    q   = u[1,:]
    etahat = fft(eta)
    qhat = fft(q)
    
    eta_x = np.real(ifft(1j*xi*etahat))
    q_x = np.real(ifft(1j*xi*qhat))
    csq = g/H1
    
    deta = -q_x

    dq = g/H1*eta_x + delta*H2/H1*(csq*eta*eta_x + 2*q*q_x) \
            + delta**2*(alpha1*eta*q*q_x + alpha2*q*q*eta_x ) \
            + delta**2*g*alpha3*eta*eta*eta_x
    if params['order4']:
        eta_xx = np.real(ifft(-xi**2*etahat))
        eta_xxx = np.real(ifft(-1j*xi**3*etahat))
        q_xx = np.real(ifft(-xi**2*qhat))
        q_xxx = np.real(ifft(-1j*xi**3*qhat))
        dq += delta**3*((alpha4/g)*q**3*q_x + alpha5*eta**2*q*q_x + alpha6*q**2*eta*eta_x + g*alpha7*eta**3*eta_x)
        dq += delta**3*(alpha8*(q_x*q_xx + csq*eta*eta_xxx)  + alpha9*(5*csq*eta_x*eta_xx + + 2*q*q_xxx))

    csq = g/H1
    dq = -dq
    dqhat = fft(dq)
    if params['order5']:
        nu1 = params['nu1']
        nu2 = params['nu2']
        dq += params['beta1']*q**4*eta_x + params['beta2']*eta**4*eta_x + params['beta3']*q**2*eta**2*eta_x \
                + params['beta15']*q**2*(eta_x)**2 + params['beta5']*(eta_x)**3 + params['beta6']*eta*eta_x*eta_xx \
                + params['beta7']*eta**2*eta_xxx + params['beta12']*q**2*eta_xxx + params['beta4']*q**3*eta*q_x \
                + params['beta9']*q*eta**3*q_x + params['beta10']*q*q_x*eta_xx + params['beta11']*eta_x*q_x**2 \
                + params['beta16']*(q**2*q_x**2-eta_x**2*q**2/csq) + params['beta8']*q*eta_x*q_xx + params['beta13']*eta*q_x*q_xx \
                + params['beta14']*q*eta*q_xxx
        if params['order7'] == False:
            # 5th-order accurate
            dq = np.real(ifft(dqhat/( 1 - xi**2*delta**2*(-mu) + xi**4*delta**4*(nu1 + nu2 - mu**2) )))
        else: # All 5th-order terms + linear 7th-order terms
            nu3 = params['nu3']
            dq = np.real(ifft(dqhat/( 1 - xi**2*delta**2*(-mu) + xi**4*delta**4*(nu1 + nu2 - mu**2) - xi**6*delta**6*nu3 )))
    else: # 3rd-order accurate
        dq = np.real(ifft(dqhat/( 1 - xi**2*delta**2*(-mu) )))

    du[0,:] = deta
    du[1,:] = dq
    return du


def homogenized_coefficients(H,bathy,b_amp=None,delta=1.0,eps=1.e-10,order5=False,order7=False):
    """
    Computes homogenized coefficients for a given periodic bathymetry profile.

    Parameters:
    H (function): A function that takes a single argument (y) and returns the unperturbed water depth at that point.
    bathy (str): The type of bathymetry. Can be either 'pwc' (piecewise-constant) or 'smooth'.
    b_amp (float, optional): The amplitude of the bathymetry. Required if bathy is 'pwc'.
    delta (float, optional): The period of the bathymetry.
    eps (float, optional): The desired accuracy of the numerical integration.

    Returns:
    dict: A dictionary containing the homogenized coefficients.
    """
    params = {}

    params['order5'] = order5
    params['order7'] = order7

    if bathy == 'pwc':
        # piecewise-constant bathymetry
        # Use exact formulas because numerical quadrature struggles
        assert(b_amp is not None)
        hA = 1
        hB = 1-b_amp
        params['H1'] = (hA+hB)/(2*hA*hB)
        params['H2'] = (hA**2+hB**2)/(2*hA**2*hB**2)
        params['H3'] = (hA**3+hB**3)/(2*hA**3*hB**3)
        params['H4'] = (hA**4+hB**4)/(2*hA**4*hB**4)
        params['H5'] = (hA**5+hB**5)/(2*hA**5*hB**5)
        params['H6'] = (hA**6+hB**6)/(2*hA**6*hB**6)
        params['H7'] = (hA**7+hB**7)/(2*hA**7*hB**7)
        C3 = -(1/hA-1/hB)**2/192
        C11 = (1/hA+1/hB)*C3
    else:
        params['H1'] = quad(lambda y: 1/H(y),0,delta,epsabs=eps,epsrel=eps)[0]
        params['H2'] = quad(lambda y: 1/H(y)**2,0,delta,epsabs=eps,epsrel=eps)[0]
        params['H3'] = quad(lambda y: 1/H(y)**3,0,delta,epsabs=eps,epsrel=eps)[0]
        params['H4'] = quad(lambda y: 1/H(y)**4,0,delta,epsabs=eps,epsrel=eps)[0]
        params['H5'] = quad(lambda y: 1/H(y)**5,0,delta,epsabs=eps,epsrel=eps)[0]
        params['H6'] = quad(lambda y: 1/H(y)**6,0,delta,epsabs=eps,epsrel=eps)[0]
        params['H7'] = quad(lambda y: 1/H(y)**7,0,delta,epsabs=eps,epsrel=eps)[0]

        ih1 = lambda y: 1/H(y)
        b1 = bracket(ih1)
        C3f = lambda y: b1(y)**2
        C3 = -quad(C3f,0,1,epsabs=eps,epsrel=eps)[0]

        ih2 = lambda y: 1/H(y)**2
        b2 = bracket(ih2)
        integrand = lambda y: b1(y)*b2(y)
        C11 = -quad(integrand,0,1,epsabs=eps,epsrel=eps)[0]


    H1, H2, H3, H4, H5, H6, H7 = params['H1'], params['H2'], params['H3'], params['H4'], params['H5'], params['H6'], params['H7']

    params['delta'] = delta
    params['C11'] = C11
    params['C3'] = C3
    params['mu'] = -params['C3']/H1**2
    params['gam'] = -params['C11']/H1**2

    params['alpha1'] = 2*(H2**2-2*H1*H3)/H1**2
    params['alpha2'] = (3*H2**2-2*H1*H3-3*H4)/(2*H1**2)
    params['alpha3'] = (H2**2-H1*H3)/H1**3
    params['alpha4'] = (3*H2**3 - 4*H1*H2*H3              - 3*H2*H4 + 4*H1*H5)/H1**2
    params['alpha5'] = 2*(H2**3 - 3*H1*H2*H3 + 3*H1**2*H4                    )/H1**3
    params['alpha6'] = (3*H2**3 - 7*H1*H2*H3 + 3*H1**2*H4 - 3*H2*H4 + 6*H1*H5)/H1**3
    params['alpha7'] =   (H2**3 - 2*H1*H2*H3 +   H1**2*H4                    )/H1**4
    params['alpha8'] = 2*(params['mu']*H2/H1-params['gam'])
    params['alpha9'] = params['mu']*H2/H1
    params['alpha10'] = (2*C11*H1 - 3*C3*H2)/H1**4
    params['alpha11'] = -4*C3*H2/H1**3

    if order5:
        if bathy == 'pwc':
            gam1 = 1/hA
            gam2 = 1/hB
            C1b2b2 = (gam1-gam2)**4*(gam1+gam2)**3/73728
            C3b1b1 = (gam1-gam2)**2*(gam1**3+gam2**3)/384
            C2b1b2 = (gam1**5-gam2*gam1**4-gam1*gam2**4+gam2**5)/384
            Cb1b4 = (gam1-gam2)*(gam1**4-gam2**4)/192
            Cb1b3 = (gam1-gam2)*(gam1**3-gam2**3)/192
            Cb2b2 = (gam1**2-gam2**2)**2/192
            C2b3 = 0
        else:
            ih3 = lambda y: 1/H(y)**3
            ih4 = lambda y: 1/H(y)**4
            b3 = bracket(ih3)
            b4 = bracket(ih4)
            Cb1b3 = quad(lambda y: b1(y)*b3(y),0,1,epsabs=eps,epsrel=eps)[0]
            Cb1b4 = quad(lambda y: b1(y)*b4(y),0,1,epsabs=eps,epsrel=eps)[0]
            Cb2b2 = quad(lambda y: b2(y)**2,0,1,epsabs=eps,epsrel=eps)[0]
            C1b2b2 = quad(lambda y: ih1(y)*b2(y)**2,0,1,epsabs=eps,epsrel=eps)[0]
            C2b1b2 = quad(lambda y: ih1(y)*b1(y)*b2(y),0,1,epsabs=eps,epsrel=eps)[0]
            C3b1b1 = quad(lambda y: ih3(y)*b1(y)**2,0,1,epsabs=eps,epsrel=eps)[0]
            C2b3 = quad(lambda y: ih2(y)*b3(y),0,1,epsabs=eps,epsrel=eps)[0]

            bb1 = bracket(b1)
            params['nu1'] = quad(lambda y: bb1(y)**2/H(y),0,1,epsabs=eps,epsrel=eps)[0]/H1**3
            params['nu2'] = 3*quad(lambda y: bb1(y)**2,0,1,epsabs=eps,epsrel=eps)[0]/H1**2

        alpha12 = (5*H3**2+6*H2*H4-6*H5*H1-15*H6)/(g*H1)
        alpha13 = (-9*H2**2*H3/2-2*H3**2*H1+3*H2*H4*H1-3*H3*H4+15*H2*H5-3*H6*H1-15*H7/2)/(2*g**2*H1)
        alpha14 = (2*Cb1b3*H1+C1b2b2+2*C2b1b2+C3b1b1+3*Cb1b4+3*C11*H2)/(g**2*H1)
        alpha15 = 4*alpha14
        alpha16 = (4*C1b2b2+3*C2b1b2-4*C3b1b1+24*0+25*Cb1b4+17*C11*H2)/(g**2*H1)
        alpha17 = (Cb1b3*H1+C1b2b2+2*C2b1b2+C3b1b1+3*Cb1b4+3*C11*H2-C3*H3)/(g**2*H1)
        alpha18 = 0
        alpha19 = (-Cb1b3-Cb2b2)/(g*H1)
        alpha20 = (4*H3**2+6*H2*H4-20*H6)/(g*H1)
        alpha21 = (2*Cb1b3+Cb2b2)/(g*H1)
        alpha22 = (H6-H3**2)/(g*H1)
        alpha23 = (4*Cb1b3+Cb2b2)/(g*H1)
        
        params['beta1'] = -alpha22 + g*alpha13/H1 + (9*H2**4/4-3*H2**2*H3*H1+H3**2*H1**2-9*H2**2*H4/2+3*H3*H4*H1+9*H4**2/4)/(g*H1**3)
        params['beta2'] = g*(H2**4/H1**3-3*H2**2*H3/H1**2+H3**2/H1+2*H2*H4/H1-H5)/(H1**2)
        params['beta3'] = g*alpha12/H1+(9*H2**4/2-16*H2**2*H3*H1+2*H3**2*H1**2+6*H2*H4*H1**2-9*H2**2*H4/2+3*H3*H4*H1+12*H2*H5*H1)/H1**4
        params['beta4'] = alpha20+(6*H2**4-22*H2**2*H3*H1+4*H3**2*H1**2+6*H2*H4*H1**2-6*H2**2*H4+6*H3*H4*H1+16*H2*H5*H1)/(g*H1**3)
        params['beta5'] = g**3*alpha14/H1**3-g**2*alpha23/H1**2+g*(-8*C3*H2**2/H1+2*C3*H3+3*C3*H4/H1)/(H1**4)
        params['beta6'] = g*(16*C11*H2-26*C3*H2**2/H1+10*C3*H3)/(H1**4)
        params['beta7'] = g**2*alpha21/H1**2+g*(6*C11*H2-5*C3*H2**2/H1+2*C3*H3)/(H1**4)
        params['beta8'] = g**2*alpha15/H1**2-2*g*alpha23/H1+(-4*Cb1b3*H1**2-27*C3*H2**2+6*C3*H3*H1+9*C3*H4)/(H1**4)
        params['beta9'] = (2*H2**4/H1**2-8*H2**2*H3/H1+4*H3**2+8*H2*H4-8*H5*H1)/(H1**2)
        params['beta10'] = -2*g*alpha21/H1 + (8*C11*H2-28*C3*H2**2/H1+12*C3*H3)/(H1**3)
        params['beta11'] = g*alpha21/H1+(12*C11*H2-22*C3*H2**2/H1+10*C3*H3)/(H1**3)
        params['beta12'] = g**2*alpha17/H1**2+g*alpha19/H1+(-2*Cb1b3-7*C3*H2**2/H1**2+2*C3*H3/H1+3*C3*H4/H1**2)/(H1**2)
        params['beta13'] = 4*g*alpha21/H1+(28*C11*H2-24*C3*H2**2/H1+8*C3*H3)/H1**3
        params['beta14'] = (8*C11*H2-10*C3*H2**2/H1+4*C3*H3)/H1**3
        params['beta15'] = (-2*C2b3-g**2*alpha14)/H1**2
        params['beta16'] = C2b3/(g*H1)

    if params['order5']:
        if bathy == 'pwc':
            assert(b_amp==0.7)
            params['nu1'] = 1.5e-3/params['H1']**3
            params['nu2'] = 3*7.1e-4/params['H1']**2
        elif bathy == 'sinusoidal':
            pass # These are already computed in the function above
            #params['nu1'] = 2.6e-3/params['H1']**3
            #params['nu2'] = 3*9.4e-4/params['H1']**2
    if params['order7']:
        if bathy == 'pwc':
            assert(b_amp==0.7)
            params['nu3'] = -6.164935733929244e-05
        elif bathy == 'sinusoidal':
            assert(b_amp==0.8)
            params['nu3']=-8.199369555627172e-05

    return params


def solve_SWH(h_amp=0.3, b_amp=0.5, width=5.0, bathy='pwc',IC='pulse',L=200,tmax=100.,m=256, dtfac=0.05,
                order4=True,order5=False,order7=False,make_anim=True,skip=128,num_plots=100):
    """
    Solve the homogenized SW equations using Fourier spectral collocation in space
    and SSPRK3 in time, on the domain (-L/2,L/2).

    Parameters:
    -----------
    h_amp : float, optional
        Amplitude of the initial wave. Default is 0.3.
    b_amp : float, optional
        Amplitude of the bathymetry. Default is 0.5.
    width : float, optional
        Width of the initial wave. Default is 5.0.
    bathy : str, optional
        Type of bathymetry. Can be 'sinusoidal', 'tanh', or 'pwc' (piecewise-constant). Default is 'pwc'.
    IC : str, optional
        Type of initial condition. Can be 'pulse', 'step', or 'data'. Default is 'pulse'.
    L : float, optional
        Length of the domain. Default is 200.
    tmax : float, optional
        Maximum time to run the simulation. Default is 100.
    m : int, optional
        Number of grid points. Default is 256.
    dtfac : float, optional
        Time step factor. Default is 0.05.
    order4 : bool, optional
        Whether to use fourth-order corrections. Default is True.
    order5 : bool, optional
        Whether to use fifth-order corrections. Default is False.
    make_anim : bool, optional
        Whether to create an animation of the simulation. Default is True.
    skip : int, optional
        Number of data points to skip when using 'data' initial condition. Default is 128.

    Returns:
    --------
    x : numpy.ndarray
        Array of grid points.
    xi : numpy.ndarray
        Array of wavenumbers.
    momentum : list
        List of arrays of momentum at each time step.
    eta : list
        List of arrays of surface elevation at each time step.
    anim : matplotlib.animation.FuncAnimation or None
        Animation of the simulation, if make_anim is True. Otherwise, None.
    """
    # ================================
    # Compute homogenized coefficients
    # ================================

    delta = 1.0
    if bathy == 'sinusoidal':
        b = lambda y: b_amp/2 * np.sin(2*np.pi*y) - 1. + b_amp/2
    elif bathy == 'tanh':
        s = 1000 # Smoothing parameter
        b = lambda y: -1+b_amp*(1+(np.tanh(s*(y-0.25))*(-np.tanh(s*(y-0.75)))))/2
    elif bathy == 'pwc': # piecewise-constant
        b = lambda y: -1 + b_amp*((y-np.floor(y))>0.25)*((y-np.floor(y))<=0.75)
    n0 = 0
    H = lambda y: n0-b(y)

    params = homogenized_coefficients(H,bathy,b_amp=b_amp,order5=order5,order7=order7)
    params['order4'] = order4
     # ================================

    # Grid
    x = np.arange(-m/2,m/2)*(L/m)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L

    from nodepy import rk
    method = rk.loadRKM('BS5').__num__()
    dt = dtfac * 1.73/np.max(xi)
    fy = np.zeros((len(method),2,m))

    
    q0 = np.zeros_like(x)
    if IC == "pulse":
        eta0 = h_amp * np.exp(-x**2 / width**2)
    elif IC == "step":
        eta0 = h_amp*(x<0)
    else:
        data = np.loadtxt(IC)
        data = data[::skip,:]
        x = data[:,0]
        eta0 = data[:,1]
        q0 = data[:,2]
        print(m, len(x))
        assert(m==len(x))
        L = x[-1]-x[0]+x[1]-x[0]
        xi = np.fft.fftfreq(m)*m*2*np.pi/L
        
    u = np.zeros((2,len(x)))

    u[0,:] = eta0
    u[1,:] = q0

    du = np.zeros_like(u)

    plot_interval = tmax/num_plots
    steps_between_plots = int(round(plot_interval/dt))
    dt = plot_interval/steps_between_plots
    nmax = num_plots*steps_between_plots

    fig = plt.figure(figsize=(12,8))
    axes = fig.add_subplot(111)
    line, = axes.plot(x,u[0,:],lw=2)
    xi_max = np.max(np.abs(xi))
    axes.set_xlabel(r'$x$',fontsize=30)
    plt.close()

    eta = [u[0,:].copy()]
    momentum = [u[1,:].copy()]
    tt = [0]
    
    xi_max = np.max(np.abs(xi))

    f = IntProgress(min=0, max=100) # instantiate the bar
    display(f) # display the bar


    for n in range(1,nmax+1):
        u_new = rkm(u,xi,xxt_rhs,dt,du,fy,method,params)
            
        u = u_new.copy()
        t = n*dt

        # Plotting
        if np.mod(n,steps_between_plots) == 0:
            f.value += 1
            eta.append(u[0,:].copy())
            momentum.append(u[1,:].copy())
            tt.append(t)
        
    def plot_frame(i):
        etahat = np.fft.fft(eta[i])
        eta_spectral = spectral_representation(x[0],etahat,xi)
        x_fine = np.linspace(x[0],x[-1],5000)
        line.set_data(x_fine,eta_spectral(x_fine))
        axes.set_title('t= %.2e' % tt[i])

    if make_anim:
        anim = matplotlib.animation.FuncAnimation(fig, plot_frame,
                                           frames=len(eta), interval=100)
        anim = HTML(anim.to_jshtml())
    else:
        anim = None

    return x, xi, momentum, eta, anim


def eta_variation(x,eta,momentum,bathy,b_amp):
    m = len(x)
    L = x[-1]-x[0]
    xi = np.fft.fftfreq(m)*m*2*np.pi/L
    delta = 1
    eps = 1e-7
    if bathy == 'sinusoidal':
        b = lambda y: b_amp/2 * np.sin(2*np.pi*y) - 1. + b_amp/2
    elif bathy == 'tanh':
        s = 2000 # Smoothing parameter
        b = lambda y: -1+b_amp*(1+(np.tanh(s*(y-0.5))*(-np.tanh(s*(y-1.)))))/2
    elif bathy == 'pwc': # piecewise-constant
        b = lambda y: -1 + b_amp*((y-np.floor(y))>0.5)

    eta0 = 0.
    H = lambda y: eta0-b(y)
    ih1 = lambda y: 1/H(y)
    ih2 = lambda y: 1/H(y)**2
    b1 = bracket(ih1); b1 = np.vectorize(b1)
    b2 = bracket(ih2); b2 = np.vectorize(b2)
    if bathy == 'tanh': # should be pwc
        gam1 = 1.; gam2 = 1/(1-b_amp)
        bb1 = lambda y: ((y<=0.5)*((gam2-gam1)*y+2*y**2*(gam1-gam2)) + (y>0.5)*(-gam1+3*gam1*y-2*y**2*gam1+gam2-3*y*gam2+2*y**2*gam2))/8
        bb1 = np.vectorize(bb1)
        C3 = -(gam1-gam2)**2/192
    else:
        bb1 = bracket(b1); bb1 = np.vectorize(bb1)
        C3f = lambda y: b1(y)**2
        C3 = -quad(C3f,0,1,epsabs=eps,epsrel=eps)[0]

    H1 = quad(lambda y: 1/H(y),0,delta,epsabs=eps,epsrel=eps)[0]
    H2 = quad(lambda y: 1/H(y)**2,0,delta,epsabs=eps,epsrel=eps)[0]
    H3 = quad(lambda y: 1/H(y)**3,0,delta,epsabs=eps,epsrel=eps)[0]
    braceH2 = lambda y: 1/H(y)**2 - H2; braceH2 = np.vectorize(braceH2)
    braceH3 = lambda y: 1/H(y)**3 - H3; braceH3 = np.vectorize(braceH3)
    
    xx = x - np.floor(x)

    q = momentum
    etahat = fft(eta)
    qhat = fft(q)
    eta_x = np.real(ifft(1j*xi*etahat))
    eta_xx  = np.real(ifft(-xi**2*etahat))
    eta_xxx = np.real(ifft(-1j*xi**3*etahat))
    q_x   = np.real(ifft(1j*xi*qhat))

    eta2 = b1(xx)/H1 * eta_x - q**2 * braceH2(xx)/(2*g)
    eta2 += q**2*eta*braceH3(xx)/g - q*q_x*(2*b1(xx)*H2/H1-b2(xx))/g
    eta2 += eta*eta_x*(b1(xx)*H2/H1-b2(xx))/H1
    eta2 -= bb1(xx)*eta_xx/H1
    eta2 -= C3*b1(xx)*eta_xxx/H1**3
    return eta2
