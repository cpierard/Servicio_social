import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import animation
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Función para extraer datos del archivo hdf5

dx, dy = 0.1/256., 0.15/256.

x, y = np.mgrid[slice(0, 0.1, dx), slice(0, 0.15, dy)]

def extraer_datos(nombre_h5):

    with h5py.File(nombre_h5, 'r') as hdf:
        base_items = list(hdf.items())
        print(base_items, '\n')
        tasks = hdf.get('tasks')
        scales = hdf.get('scales')
        #scales_items = list(scales.items())
        #print(scales_items)
        #tasks_items = list(tasks.items())
        #print(tasks_items)

        T = np.array(tasks.get('T'))
        ρ = np.array(tasks.get('ρ'))
        s = np.array(tasks.get('s'))
        t = np.array(scales.get('sim_time'))
        NN = np.array(tasks.get('NN'))
        print('Exportando...')

    return T, ρ, s, t, NN

    #Función animar

def animar_dedalus(xm, ym, S, R,  t):

    fig, (ax0, ax1) = plt.subplots(ncols=2 , gridspec_kw = {'width_ratios':[1.82, 1.49]}, sharey=True)
    fig.set_size_inches(11,7)

    im1 = ax0.pcolormesh(x, y, S[0, :-1, :-1], cmap='rainbow')
    ax0.set_title('$T\ ( ^oC)$', fontsize= 20)
    ax0.set_ylim(0,0.15)
    cbar = fig.colorbar(ax0, ax=ax1)
    #ax0.set_ylabel('$z \ (cm)$', fontsize=tick_size)
    #ax0.set_xlabel('$x \ (cm)$', fontsize=tick_size)

    im2 = ax1.pcolormesh(x, y, R[0, :-1, :-1], cmap='rainbow')
    ax1.set_title('$N^2 \ (s^{-2})$', fontsize= 20)
    ax1.set_ylim(0,0.15)
    #ax0.set_ylabel('$z \ (cm)$', fontsize=tick_size)
    #ax0.set_xlabel('$x \ (cm)$', fontsize=tick_size)

    def init():
        print('update init')
        ax0.set_array(np.ravel(S[0,:-1,:-1]))
        ax1.set_array(np.ravel(R[0,:-1,:-1]))
        #tx.set_text('t = ' + str(t[0]))
        return p

    def update(frame):
        #vmin = np.min(S[frame])
        #vmax = np.max(S[frame])
        #if t[frame] < 100:
        #    time_str = str(t[frame])[:4]
        #elif t[frame] >= 100:
        #    time_str = str(t[frame])[:5]
        #elif t[frame] >= 1000:
        #    time_str = str(t[frame])[:6]

        ax0.set_array(np.ravel(S[frame, :-1, :-1]))
        ax1.set_array(np.ravel(R[frame, :-1, :-1]))
        #p.set_clim(vmin, vmax)
        #plt.title(str(t[frame]))
        #plt.xlabel('$x \ (cm)$', fontsize = 18)
        #plt.ylabel('$z \ (cm)$', fontsize = 18)
        #tx.set_text('t = ' + str(t[frame]))
        #tx.set_text('t = ' + time_str )
        #plt.title('Temperatura')

        return p

    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(0,len(S))],  blit = False)
    plt.show()
    return anim

#Abajo tienes que poner el nombre del archivo hdf5 en donde guardaste los datos.

T_dat , ρ_dat, s_dat, t_dat, NN_dat = extraer_datos('ugm_24/ugm_24.h5')

####################ANIMACION######################

#print(dy)
anima_T = animar_dedalus(x, y, T_dat[1:,:,:], NN_dat[1:,:,:], t_dat[1:])

## Video con ImageMagick
#anima_T.save('ugm_24/Temp_24_v2.mp4', writer='imagemagick', fps=24) #nombre de como quieres que se guarde el video. 'imagemagick'

## Video con FFMpeg
#mywriter = animation.FFMpegWriter()
#anima_T.save('prueba2.mp4',writer=mywriter, fps=30)
