import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools
import time

import logging
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

Lx, Ly = (0.5, 0.5)
nx, ny = (256, 256)

x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Chebyshev('y', ny, interval=(0, Ly), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['p', 'u', 'v', 'ρ', 'uy', 'vy', 'ρy'])

ρ_lower = 998.23 #Tap_water
A = 7e-4 #Atwood number
ρ_upper = (ρ_lower * (1 + A))/(1 - A) #

ρ0 = (ρ_upper + ρ_lower)/2

problem.parameters['Re'] = 1e8
problem.parameters['ρ0'] = ρ_lower
problem.parameters['g'] = 9.8
problem.parameters['ρ_upper'] = ρ_upper

problem.add_equation("dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) = - u*dx(v) - v*vy - g*(ρ - ρ0)/ρ0")
problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(ρ) = - u*dx(ρ) - v*ρy")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("ρy - dy(ρ) = 0")

problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")
problem.add_bc("left(ρ) = ρ_upper")


ts = de.timesteppers.RK443

solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
ρ = solver.state['ρ']
yb, yt = y_basis.interval

for i in range(0, int(ny/2)):
    ρ['g'][:, i] = ρ['g'][:, i] + ρ_lower

for i in range(int(ny/2), ny):
     ρ['g'][:, i] = ρ['g'][:, i] + ρ_upper

#ρ['g'][64, 64] = ρ_lower #Perturbación

# Initial timestep
dt = 0.02
# Integration parameters
solver.stop_sim_time = 2.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('RT_1', sim_dt=0.01, max_writes=50)
snapshots.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, dt, safety=0.1,  max_dt=0.02, threshold = 0.01)
CFL.add_velocities(('u', 'v'))

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = CFL.compute_dt()
    dt = solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()
logger.info('Iterations: %i' %solver.iteration)
logger.info('Sim end time: %f' %solver.sim_time)
logger.info('Run time: %.2f sec' %(end_time-start_time))
logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
