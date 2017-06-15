import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import animation

#Función para extraer datos del archivo hdf5

dx, dy = 0.2/256., 0.35/256.

x, y = np.mgrid[slice(0, 0.2, dx), slice(0, 0.35, dy)]

def extraer_datos(nombre_h5):

    with h5py.File(nombre_h5, flag ='r') as hdf:
        base_items = list(hdf.items())
        print(base_items, '\n')
        tasks = hdf.get('tasks')
        tasks_items = list(tasks.items())
        print(tasks_items)

        T = np.array(tasks.get('T'))
        print(T.shape)

        ρ = np.array(tasks.get('ρ'))
        print(ρ.shape)

    return T, ρ

    #Función animar

def animar_dedalus(xm, ym, S, CMAP):
    fig, axis = plt.subplots(figsize=(4,7))
    p = axis.pcolormesh(xm, ym, S[0,:,:], cmap=CMAP)
    plt.colorbar(p)

    def init():
                print('update init')
                p.set_array(np.ravel(S[0,:-1,:-1]))
                return p

    def update(frame):
        p.set_array(np.ravel(S[frame, :-1, :-1]))
        plt.title(frame)
        return p

    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(1,len(S))], init_func=init,  blit = False)
    plt.show()
    return anim

#Abajo tienes que poner el nombre del archivo hdf5 en donde guardaste los datos.

T_dat , ρ_dat = extraer_datos('strat_conv_analisys/strat_conv_analisys_s3.h5')

anima_T = animar_dedalus(x, y, T_dat, 'rainbow')
#mywriter = animation.FFMpegWriter()
#anima_T.save('strat_conv_T.mp4',writer=mywriter, fps=38) #nombre de como quieres que se guarde el video.

'''
anima_ρ = animar_dedalus(x, y, ρ_dat, 'rainbow')
#mywriter = animation.FFMpegWriter()
anima_ρ.save('RB_conv_rho.mp4',writer=mywriter, fps=30) #nombre de como quieres que se guarde el video.


print(T_dat.shape)

fig, axis = plt.subplots(figsize=(4,7))
pT = axis.pcolormesh(x, y, T_dat[-1,:,:], cmap='rainbow');
plt.colorbar(pT)
plt.title('Temperature, frame 240')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''
