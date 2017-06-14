
import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import animation

Lx, Ly = (0.2, 0.35)
nx, ny = (256, 256)

dx, dy = Lx/nx, Ly/ny

x, y = np.mgrid[slice(0, Lx, dx), slice(0, Ly, dy)]

print(x.shape)
print(y.shape)

#Extraer datos HDF5

with h5py.File('analisis_2d_conv/analisis_2d_conv_s1/analisis_2d_conv_s1_p0.h5', flag ='r') as hdf:
    base_items = list(hdf.items())
    print(base_items, '\n')
    tasks = hdf.get('tasks')
    tasks_items = list(tasks.items())
    print(tasks_items)

    T_dat = np.array(tasks.get('T'))
    print(T_dat.shape)

    ρ_dat = np.array(tasks.get('ρ'))
    print(ρ_dat.shape)

#Función para animar

def animar_dedalus(xm, ym, S, CMAP):
    fig, axis = plt.subplots(figsize=(4,7))
    p = axis.pcolormesh(xm, ym, S[0,:,:], cmap=CMAP)
    plt.colorbar(p)
    plt.title('0')

    def init():
                print('update init')
                p.set_array(np.ravel(S[0,:-1,:-1]))
                plt.title('0')
                return p

    def update(frame):
        p.set_array(np.ravel(S[frame, :-1, :-1]))
        plt.title(frame)
        return p

    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(1,len(S))], init_func=init,  blit = False)
    plt.show()
    return anim

#Animación

anima = animar_dedalus(x, y, ρ_dat, 'RdBu_r')
#mywriter = animation.FFMpegWriter()
#anima.save('2d_conv.mp4',writer=mywriter, fps=30) #nombre de como quieres que se guarde el video.
