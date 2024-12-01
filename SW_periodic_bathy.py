#!/usr/bin/env python
# encoding: utf-8

r"""
Shallow water flow
==================

Solve the one-dimensional shallow water equations including bathymetry:

.. math::
    h_t + (hu)_x & = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x & = -g h b_x.

Here h is the depth, u is the velocity, g is the gravitational constant, and b
the bathymetry.  
"""

import numpy as np
from clawpack import riemann
parallel = True

if parallel:
    from clawpack import petclaw as pyclaw
else:
    from clawpack import pyclaw

def b4step(solver,state):
    if state.t>state.problem_data['t_periodic'] and solver.bc_lower[0]==pyclaw.BC.wall:
        solver.bc_lower[0]=2
        solver.bc_upper[0]=2
        solver.aux_bc_lower[0]=2
        solver.aux_bc_upper[0]=2

def setup(solver_type='classic', riemann_solver='geoclaw',weno_order=5,
          outdir='./_output', h_amp=0.025, b_amp=0.7, bathy='pwc', width=3.,
          u_l=0., IC='pulse', t_periodic=1e10, xupper=400., mx=1000, tfinal=180, nout=100):


    if riemann_solver == 'geoclaw':
        rp = riemann.sw_aug_1D
    elif riemann_solver == 'fwave':
        rp = riemann.shallow_bathymetry_fwave_1D
    else:
        raise Exception

    if solver_type == 'classic':
        solver = pyclaw.ClawSolver1D(rp)
        solver.limiters = pyclaw.limiters.tvd.vanleer
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver1D(rp)
        solver.weno_order = weno_order
    else:
        raise Exception
    solver.fwave = True
    solver.num_waves = 2
    solver.num_eqn = 2
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    #solver.before_step = b4step 

    xlower = 0.0
    x = pyclaw.Dimension( xlower, xupper, mx, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 2, 1)

    # Gravitational constant
    state.problem_data['grav'] = 9.8
    state.problem_data['dry_tolerance'] = 1e-3
    state.problem_data['sea_level'] = 0.0
    state.problem_data['t_periodic'] = t_periodic

    xc = state.grid.x.centers
    if bathy == 'sinusoidal':
        state.aux[0, :] = b_amp/2 * np.sin(2*np.pi*xc) - 1. + b_amp/2
    elif bathy == 'pwc': # piecewise-constant
        xfrac = xc-np.floor(xc)
        state.aux[0,:] = -1 + b_amp*(xfrac>=0.5)
    elif bathy == 'constant':  # constant depth, chosen to have same linearized sound speed
        H1 = 2/(1+1/(1-b_amp))
        state.aux[0,:] = -H1

    if IC == 'pulse':
        state.q[0, :] = h_amp * np.exp(-(xc - 0)**2 / width**2) - state.aux[0, :]
        state.q[1, :] = 0.0
    elif IC == 'step':
        state.q[0,:] = h_amp*(xc<(xlower + 0.5*(xupper-xlower))) - state.aux[0,:]
        state.q[1, :] = (u_l*state.q[0,:])*(xc<(xlower + 0.5*(xupper-xlower)))
    else:
        # IC should be a text file with 3 columns: x, h, hu
        # We assume it's a wave that should travel to the right
        # so we place it at the left edge of the domain.
        data = np.loadtxt(IC)
        iclen = data.shape[0]
        if xc[0]<1:  # Check we're really at left boundary, in case of parallel runs
            state.q[0,:iclen] = data[:,1]
            state.q[0,iclen:] = 0 - state.aux[0,iclen:]
            state.q[1,:iclen] = data[:,2]
            state.q[1,iclen:] = 0
        else:
            state.q[0,:] = 0 - state.aux[0,:]
            state.q[1,:] = 0
        state.grid.add_gauges([(100.25,)])

    claw = pyclaw.Controller()
    if parallel:
        claw.keep_copy = False
    else:
        claw.keep_copy = True
    claw.tfinal = tfinal
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.setplot = setplot
    claw.write_aux_init = True
    claw.num_output_times = nout
    claw.solver.max_steps = mx

    if outdir is not None:
        claw.outdir = outdir
    else:
        claw.output_format = None


    return claw


#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Plot variables
    def bathy(current_data):
        return current_data.aux[0, :]

    def eta(current_data):
        return current_data.q[0, :] + bathy(current_data)

    def velocity(current_data):
        return current_data.q[1, :] / current_data.q[0, :]

    rgb_converter = lambda triple: [float(rgb) / 255.0 for rgb in triple]

    # Figure for depth
    plotfigure = plotdata.new_plotfigure(name='Depth', figno=0)

    # Axes for water depth
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-1.0, 1.0]
    plotaxes.ylimits = [-1.1, 0.2]
    plotaxes.title = 'Water Depth'
    plotaxes.axescmd = 'subplot(211)'

    plotitem = plotaxes.new_plotitem(plot_type='1d_fill_between')
    plotitem.plot_var = eta
    plotitem.plot_var2 = bathy
    plotitem.color = rgb_converter((67,183,219))

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = bathy
    plotitem.color = 'k'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = eta
    plotitem.color = 'k'

    # Axes for velocity
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = [-1.0, 1.0]
    plotaxes.ylimits = [-0.5, 0.5]
    plotaxes.title = 'Velocity'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = velocity
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}
    
    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
