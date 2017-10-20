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
    fig, axis = plt.subplots(figsize=(5,7.5))
    p = axis.pcolormesh(xm, ym, S[0,:,:],  norm= colors.PowerNorm(gamma=norma), cmap=CMAP)
    plt.colorbar(p)
    tx = axis.set_title(str(t[0]))
    plt.ylim(0,0.15)

    def init():
        print('update init')
        p.set_array(np.ravel(S[0,:-1,:-1]))
        tx.set_text('t = ' + str(t[0]))
        return p

    def update(frame):
        vmin = np.min(S[frame])
        vmax = np.max(S[frame])
        p.set_array(np.ravel(S[frame, :-1, :-1]))
        p.set_clim(vmin, vmax)
        #plt.title(str(t[frame]))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        tx.set_text('t = ' + str(t[frame]))
        #plt.title('Temperatura')

        return p

    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(0,len(S))],  blit = False)
    plt.show()
    return anim

#Abajo tienes que poner el nombre del archivo hdf5 en donde guardaste los datos.

T_dat , ρ_dat, s_dat, t_dat, NN_dat = extraer_datos('ugm_28/ugm_28.h5')
print('sim t')
print(t_dat.shape)


#max_v = v_dat = v_dat[-1, :, :].max()
#print(max_v)

#print(dy)
anima_T = animar_dedalus(x, y, NN_dat[1:,:,:], t_dat, 1, 'rainbow')
#mywriter = animation.FFMpegWriter()
anima_T.save('ugm_28/BV_ugm_28.mp4',writer='imagemagick', fps=24) #nombre de como quieres que se guarde el video. 'imagemagick'
#anima_T.save('prueba2.mp4',writer=mywriter, fps=30)
'''
anima_ρ = animar_dedalus(x, y, ρ_dat, 'rainbow_r')
#mywriter = animation.FFMpegWriter()
anima_ρ.save('RB_conv_rho.mp4',writer=mywriter, fps=30) #nombre de como quieres que se guarde el video.


print(T_dat.shape)

fig, axis = plt.subplots(figsize=(4,7))
pT = axis.pcolormesh(x, y, T_dat[-1,:,:], norm= colors.PowerNorm(gamma=1./2.), cmap='rainbow');
plt.annotate('hola', xy = (0.15, 0.3), xytext = (0.15, 0.32))
plt.colorbar(pT)
plt.title('Temperature, frame 240')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''
