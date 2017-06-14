import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools
import time

import logging
logger = logging.getLogger(__name__)


# ## Dominio del problema

Lx, Ly = (0.2, 0.35)
nx, ny = (256, 256)

ν = 1.8e-6
T0 = 4.5
ρ0 = 999.9699
α = 8.1e-6
Prandtl = 1. #ν/α #0.22
Rayleigh = 5.8e7


x_basis = de.Fourier('x', nx, interval=(0, Lx))
y_basis = de.Chebyshev('y', ny, interval=(0, Ly))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)


# ## Ecuaciones
problem = de.IVP(domain, variables=['p', 'u', 'v', 'ρ', 'T', 'uy', 'vy', 'Ty'])

problem.meta['p', 'T', 'u', 'v', 'ρ']['y']['dirichlet'] = True

problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
#problem.parameters['F'] = F = 1
problem.parameters['ρ0'] = ρ0
problem.parameters['T_0'] = T0 #10ºC
problem.parameters['K'] = 1.3e-7
problem.parameters['g'] = 9.8
problem.parameters['α'] = α

problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dy(uy)) + dx(p) = -(u*dx(u) + v*uy)")
problem.add_equation("dt(v) - R*(dx(dx(v)) + dy(vy)) + dy(p) = -(u*dx(v) + v*vy) + g*(ρ - ρ0)/ρ0")
problem.add_equation("ρ = ρ0 - ρ0*α*(T - T_0)")
problem.add_equation("dt(T) - K*(dx(dx(T)) + dy(Ty)) = - u*dx(T) - v*Ty")
problem.add_equation("Ty - dy(T) = 0")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")

problem.add_bc("left(T) = 5.0")
problem.add_bc("right(T) = 4.0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

solver = problem.build_solver(de.timesteppers.RK222)


# ## Condiciones iniciales

x = domain.grid(0)
y = domain.grid(1)
T = solver.state['T']
Ty = solver.state['Ty']
ρ = solver.state['ρ']

yb, yt = y_basis.interval

x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)

a,b = T['g'].shape
T['g'] = np.random.rand(a,b)*1e-1 + 4.5
ρ['g'] = ρ0 - ρ0*α*(T['g'] - T0)

# Initial timestep
dt = 0.125
# Integration parameters
solver.stop_sim_time = 30
solver.stop_wall_time = 30 * 70.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('RB_conv_analysis', sim_dt=0.25, max_writes=300)
snapshots.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'v'))

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v) / R", name='Re')


# In[23]:

try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        #print("hola")
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            # Update plot of scalar field
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
