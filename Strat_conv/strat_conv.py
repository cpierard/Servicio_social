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
k = 2e-5
T0 = 4.0
T_b = 8.0# 0.0
g = 9.8
κ = 1.3e-7
ρ0 = 999.9720 # densidad a 4ºC
α = 8.1e-6
T_air = 21.
T_top = 4.0 #8.
z_int = 0.18

Prandtl = ν/κ
print(Prandtl)
Rayleigh = (g*α*T0**2*(0.22)**3)/(ν*κ)
print(Rayleigh)

x_basis = de.Fourier('x', nx, interval=(0, Lx))
y_basis = de.Chebyshev('y', ny, interval=(0, Ly))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)


# ## Ecuaciones

problem = de.IVP(domain, variables=['p', 'u', 'v', 'ρ', 'T', 'uy', 'vy', 'Ty'])

problem.meta['p', 'T', 'u', 'v', 'ρ']['y']['dirichlet'] = True

problem.parameters['ν'] = ν
problem.parameters['κ'] = κ
problem.parameters['T_air'] = T_air
problem.parameters['k'] = k
problem.parameters['ρ0'] = ρ0
problem.parameters['T_0'] = T0 #4.0 ºC
problem.parameters['g'] = 9.8
problem.parameters['α'] = α
problem.parameters['T_b'] = T_b
problem.parameters['T_top'] = T_top

problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(u) - ν*(dx(dx(u)) + dy(uy)) + dx(p) = -(u*dx(u) + v*uy)")
problem.add_equation("dt(v) - ν*(dx(dx(v)) + dy(vy)) + dy(p) = -(u*dx(v) + v*vy) - g*(ρ - ρ0)/ρ0")
problem.add_equation("ρ = ρ0 - ρ0*α*(T - T_0)**2")
problem.add_equation("dt(T) - κ*(dx(dx(T)) + dy(Ty)) = - u*dx(T) - v*Ty - k*(T - T_air)")
problem.add_equation("Ty - dy(T) = 0")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")

problem.add_bc("left(T) = T_b")
problem.add_bc("right(T) = T_top")
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

a, b = T['g'].shape
pert =  np.random.rand(nx,ny) * (yt - y) * (y - 0.18) * y * (y - 0.26) * 1000

T['g'] = 40.257128492422666*y - 300.5817711700071*y**2 + 1113.2658191481735*y**3
T['g'] = T['g'] + pert

ρ['g'] = ρ0 - ρ0*α*(T['g'] - T0)**2

# Initial timestep
dt = 0.02
# Integration parameters
solver.stop_sim_time = 80
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('strat_conv_analysis', sim_dt=0.25, max_writes=400)
snapshots.add_system(solver.state)

# CFL
#CFL = flow_tools.CFL(solver, initial_dt = dt, max_change = 0.5)
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.1, max_change=1.5, min_change=0.5, max_dt=0.02, threshold=0.01)
CFL.add_velocities(('u', 'v'))

#Solver

try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            # Update plot of scalar field
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
