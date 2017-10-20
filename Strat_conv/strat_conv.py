import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools
import time

import logging
logger = logging.getLogger(__name__)

# ## Dominio del problema

Lx, Ly = (0.1, 0.15)
nx, ny = (256, 256)

ν = 1.8e-6 # m^2/s viscocidad cinemática
k = 2e-5 # s^-1
g = 9.8 #  m/s^2
κ = 1.3e-7 #thermal difussivity m^2/s

ρ0 = 1006.75  # kg/m^3 promedio de las densidades iniciales
α = 0.15 # kg/m^3ºC thermal expansion
β = 0.78 # kg/m^3%0 salinity contraction
s_top = 0. #salinidad top (partes por mil)
s_bot = 12.5 #salinidad bottom (partes por mil)
s0 = 12.5

z_int = 0.08 #Altura de convección (m)

T0 = 20.0 # ºC Promedio de las temperaturas iniciales
T_b = 24.0 #Bottom temperature ºC
T_air = 20. # ºC temperatura de aire.
T_top = 20.0
L_conv = 0.35
Reynolds = 100
Schmidt = 1000
Peclet = 1e6

x_basis = de.Fourier('x', nx, interval=(0, Lx))
y_basis = de.Chebyshev('y', ny, interval=(0, Ly))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)


# ## Ecuaciones

problem = de.IVP(domain, variables=['p', 'u', 'v', 'ρ', 's', 'T', 'uy', 'vy', 'Ty', 'sy'])

problem.meta['p', 'T', 'u', 'v', 'ρ', 's']['y']['dirichlet'] = True

#problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
#problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
#problem.parameters['F'] = F = 1
problem.parameters['ν'] = ν
problem.parameters['κ'] = κ
problem.parameters['T_air'] = T_air
problem.parameters['k'] = k
problem.parameters['ρ0'] = ρ0
problem.parameters['T_0'] = T0 #4.0 ºC
problem.parameters['g'] = 9.8
problem.parameters['α'] = α
problem.parameters['β'] = β
problem.parameters['T_b'] = T_b
problem.parameters['T_top'] = T_top
problem.parameters['s0'] = s0
problem.parameters['s_top'] = s_top
problem.parameters['s_bot'] = s_bot
problem.parameters['Lx'] = Lx

problem.parameters['Re'] = Reynolds
problem.parameters['Sc'] = Schmidt
problem.parameters['Pe'] = Peclet

problem.add_equation("dx(u) + vy = 0") #continuidad
problem.add_equation("dt(u) - ν*(dx(dx(u)) + dy(uy)) + dx(p) = -(u*dx(u) + v*uy)") #N-S x
problem.add_equation("dt(v) - ν*(dx(dx(v)) + dy(vy)) + dy(p) = -(u*dx(v) + v*vy) - g*(ρ - ρ0)/ρ0") #N-S y
problem.add_equation("ρ = ρ0 - α*(T - T_0) + β*(s - s0)") #ecuación de estado

#problem.add_equation("dt(T) - κ*(dx(dx(T)) + dy(Ty)) = - u*dx(T) - v*Ty - k*(T - T_air)") #conservación energía
#problem.add_equation("dt(s) - 1/(Re*Sc)*(dx(dx(s)) + dy(sy)) = - u*dx(s) - v*sy") #ecuación para salinidad

problem.add_equation("dt(T) - 1/Pe*(dx(dx(T)) + dy(Ty)) = - u*dx(T) - v*Ty") #conservación energía
problem.add_equation("dt(s) - 1/(Pe)*(dx(dx(s)) + dy(sy)) = - u*dx(s) - v*sy") #Salinidad

problem.add_equation("Ty - dy(T) = 0")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("sy - dy(s) = 0")

problem.add_bc("left(T) = T_b")
problem.add_bc("right(T) = T_top")

problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")

problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")

problem.add_bc("left(s) = s_bot")
problem.add_bc("right(s) = s_top")

problem.add_bc("right(p) = 0", condition="(nx == 0)")

solver = problem.build_solver(de.timesteppers.RK222)


# ## Condiciones iniciales
x = domain.grid(0)
y = domain.grid(1)
T = solver.state['T']
Ty = solver.state['Ty']
ρ = solver.state['ρ']
s = solver.state['s']

def perfil_arriba(x):
    return -s_bot/(Ly - z_int)*x -(-s_bot/(Ly - z_int))*Ly

yb, yt = y_basis.interval

x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)

a, b = T['g'].shape
pert =  np.random.rand(a,b) * (yt - y) * (y - z_int) * (-y) * ((Ly - Ly/2) - y)*(0.11 - y)*(0.13 - y) * 1e7

T['g'] = np.zeros_like(y) + 20. + pert

for i in range(0, len(y[0])):
    if y[0, i] <= z_int:
        s['g'][:, i] = s_bot
    elif y[0,i] > z_int:
        s['g'][:, i] = perfil_arriba(y[0,i])


ρ['g'] = ρ0 - α*(T['g'] - T0) + β*(s['g'] - s0)

# Initial timestep
dt = 0.02
# Integration parameters
solver.stop_sim_time = 300
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('ugm_24', sim_dt=0.25, max_writes=1200)
snapshots.add_system(solver.state)
snapshots.add_task("integ(s,'x')/Lx", name='s profile')
snapshots.add_task("integ(T,'x')/Lx", name='T profile')
snapshots.add_task("integ(ρ,'x')/Lx", name='ρ profile')
snapshots.add_task("-g/ρ0*dy(ρ)", name='NN')


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
