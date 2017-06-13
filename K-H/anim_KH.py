import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import animation

#Función para extraer datos del archivo hdf5

dx, dy = 2/192., 1/96.

x, y = np.mgrid[slice(0, 2, dx), slice(0, 1, dy)]

def extraer_datos(nombre_h5):

    with h5py.File(nombre_h5, flag ='r') as hdf:
        base_items = list(hdf.items())
        print(base_items, '\n')
        tasks = hdf.get('tasks')
        tasks_items = list(tasks.items())
        print(tasks_items)

        s = np.array(tasks.get('s'))
        print(s.shape)

        s_profile= np.array(tasks.get('s profile'))
        print(s_profile.shape)

        u = np.array(tasks.get('u'))
        print(u.shape)

    return s, s_profile, u

    #Función animar

def animar_dedalus(xm, ym, S, CMAP):
    fig, axis = plt.subplots(figsize=(10,5))
    p = axis.pcolormesh(xm, ym, S[0,:,:], cmap=CMAP)

    def init():
                print('update init')
                p.set_array(np.ravel(S[0,:-1,:-1]))
                return p

    def update(frame):
        p.set_array(np.ravel(S[frame, :-1, :-1]))
        return p

    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(1,len(S))], init_func=init,  blit = False)
    plt.show()
    return anim

#Abajo tienes que poner el nombre del archivo hdf5 en donde guardaste los datos.

s_dat , s_rofile_dat, u_dat = extraer_datos('nombre/nombre_s1/nombre_s1_p0.h5')

anima_s = animar_dedalus(x, y, s_dat, 'Spectral')
mywriter = animation.FFMpegWriter()
anima_s.save('KH_prueba_mpi.mp4',writer=mywriter, fps=30) #nombre de como quieres que se guarde el video.
