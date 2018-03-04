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

def animar_dedalus(xm, ym, S, t, norma,  CMAP):
    fig, axis = plt.subplots(figsize=(6,7.5))
    fig.suptitle('$N^2 \ (s^{-2})$', fontsize = 14, fontweight='bold', x = 0.8, y = 0.95) # 'T \ ($^oC$)' #'$N^2 \ (s^{-2})$'
    #fig.suptitle('$N^2 \ (s^{-2})$', fontsize = 14, fontweight='bold')
    p = axis.pcolormesh(xm, ym, S[0,:,:], norm= colors.Normalize(vmin=-1.15,vmax=1.5), cmap=CMAP) #, vmin = -1.15, vmax = 1.5
    plt.colorbar(p)
    #p.set_clim(-1.15, 1.5) #Para plotear brunt-vaisala
    tx = axis.set_title(str(t[0]))
    plt.ylim(0,0.15)

    def init():
        print('update init')
        p.set_array(np.ravel(S[0,:-1,:-1]))
        tx.set_text('t = ' + str(t[0]))
        return p

    def update(frame):
        #vmin = np.min(S[frame])
        #vmax = np.max(S[frame])
        if t[frame] < 100:
            time_str = str(t[frame])[:4]
        elif t[frame] >= 100:
            time_str = str(t[frame])[:5]
        elif t[frame] >= 1000:
            time_str = str(t[frame])[:6]

        p.set_array(np.ravel(S[frame, :-1, :-1]))
        #p.set_clim(vmin, vmax)
        #plt.title(str(t[frame]))
        plt.xlabel('$x \ (m)$', fontsize = 18)
        plt.ylabel('$z \ (m)$', fontsize = 18)
        #tx.set_text('t = ' + str(t[frame]))
        tx.set_text('t = ' + time_str )
        #plt.title('Temperatura')

        return p

    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(0,len(S))],  blit = False)
    plt.show()
    return anim

#Abajo tienes que poner el nombre del archivo hdf5 en donde guardaste los datos.

T_dat , ρ_dat, s_dat, t_dat, NN_dat = extraer_datos('ugm_28.h5')

####################ANIMACION######################

#print(dy)
anima_T = animar_dedalus(x, y, NN_dat[1:,:,:], t_dat[1:], 1, 'Blues')

## Video con ImageMagick
#anima_T.save('NN_28.mp4' fps=24) #nombre de como quieres que se guarde el video. 'imagemagick'

## Video con FFMpeg
Writer = animation.writers['ffmpeg']
writer = Writer(fps=24,  bitrate=1800)
anima_T.save('NN_28.mp4',writer=writer)
