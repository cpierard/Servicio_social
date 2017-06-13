import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools
import time

import logging
logger = logging.getLogger(__name__)

#Aspect ratio 2
Lx, Ly = (2., 1.)
nx, ny = (192, 96)

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Chebyshev('y',ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

#Ecuaciones y condiciones de frontera

problem = de.IVP(domain, variables=['p','u','v','uy','vy','s','sy'])

Reynolds = 1e4
Schmidt = 1.

problem.parameters['Re'] = Reynolds
problem.parameters['Sc'] = Schmidt

problem.add_equation("dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) = - u*dx(v) - v*vy")
problem.add_equation("dx(u) + vy = 0")

problem.add_equation("dt(s) - 1/(Re*Sc)*(dx(dx(s)) + dy(sy)) = - u*dx(s) - v*sy")

problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("sy - dy(s) = 0")

problem.add_bc("left(u) = 0.5")
problem.add_bc("right(u) = -0.5")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("integ(p,'y') = 0", condition="(nx == 0)")
problem.add_bc("left(s) = 0")
problem.add_bc("right(s) = 1")

#Timestepping

ts = de.timesteppers.RK443

solver =  problem.build_solver(ts)

#Condiciones iniciales

x = domain.grid(0)
y = domain.grid(1)
u = solver.state['u']
uy = solver.state['uy']
v = solver.state['v']
vy = solver.state['vy']
s = solver.state['s']
sy = solver.state['sy']

a = 0.05
sigma = 0.2
flow = -0.1
amp = -0.2

#Velocidades horizontales

u['g'] = flow*np.tanh(y/a)      #Perfil Ui_tanh
#u['g'] = flow*np.abs(y) + 0.5    #Perfil Ui_punta
#u['g'] = flow * a/y   #Perfil Ui_hiperbola
#u['g'] = flow * a/np.abs(y)  #Perfil Ui_abs_hiperb

'''
#Perfil Ui_half_shear
#dx_g, dy_g = 2/192., 1/96.
#x_g, y_g = np.mgrid[slice(0, 2, dx_g), slice(0, 1, dy_g)]
u_prueba = np.zeros_like(y)


for j in range(0, 96):

    if j < 48:
        u_prueba[0, j] = y[0, j] * 0.5

    elif j >= 48:
        u_prueba[0,j] =  y[0,j] *0.5/2 - y[0, 48] *0.5

u['g'] = u_prueba

########
'''


v['g'] = amp*np.sin(2.0*np.pi*x/Lx)*np.exp(-(y*y)/(sigma*sigma))
s['g'] = 0.5*(1+np.tanh(y/a))
u.differentiate('y',out=uy)
v.differentiate('y',out=vy)
s.differentiate('y',out=sy)

#Tiempo de simulación y CFL

solver.stop_sim_time = 2.01 #Tiempo de simulación
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.2*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v'))

#Análisis: aqui defines nombre y las variable quieres guardar en el archivo.hdf5

nombre_hdf5 = 'prueba_mpi'

analysis = solver.evaluator.add_file_handler( nombre_hdf5 , sim_dt=0.1, max_writes=100) # "nombre_archivo_hdf5"
analysis.add_task('s') #variables a guardar
analysis.add_task('u')
solver.evaluator.vars['Lx'] = Lx
analysis.add_task("integ(s,'x')/Lx", name='s profile')

#Main loop

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
