import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import animation
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Función para extraer datos del archivo hdf5

dx, dy = 0.2/256., 0.35/256.

x, y = np.mgrid[slice(0, 0.2, dx), slice(0, 0.35, dy)]

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
        print(T.shape)

        ρ = np.array(tasks.get('ρ'))
        print(ρ.shape)

        t = np.array(scales.get('sim_time'))
        print('sim t: ' +str(t.shape))

    return T, ρ, t

    #Función animar

def animar_dedalus(xm, ym, S, t, norma,  CMAP):
    #fig, axis = plt.subplots(figsize=(4,7))
    #p = axis.pcolormesh(xm, ym, S[0,:,:],  norm= colors.PowerNorm(gamma=norma), cmap=CMAP)
    #plt.colorbar(p)
    fig = plt.figure(figsize=(4,7))
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    im = ax.pcolormesh(xm, ym, S[0,:,:], cmap='rainbow')
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')
    #def init():
    #    print('update init')
    #    im.set_array(np.ravel(S[0,:-1,:-1]))

    #    return im

    def update(frame):
<<<<<<< HEAD
        p.set_array(np.ravel(S[frame, :-1, :-1]))
        plt.title(str(t[frame]))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
=======
        vmax = np.max(S[frame])
        vmin = np.min(S[frame])
        im.set_array(np.ravel(S[frame, :-1, :-1]))
        #im.set_clim(vmin, vmax)
        tx.set_text('Frame {0}'.format(i))

        #plt.title(str(t[frame]))
        #plt.xlabel('$x$')
        #plt.ylabel('$y$')
>>>>>>> bc73793d423ba1e5476ab6cd85837a51c71bf172
        #plt.title('Temperatura')

        #return im

<<<<<<< HEAD
    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(0,len(S), 1)], init_func=init,  blit = False)
=======
    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(0,len(S), 3)],  blit = False)
>>>>>>> bc73793d423ba1e5476ab6cd85837a51c71bf172
    plt.show()
    return anim

#Abajo tienes que poner el nombre del archivo hdf5 en donde guardaste los datos.

T_dat , ρ_dat, t_dat = extraer_datos('temp_salinity/temp_salinity_s6.h5')
print('sim t')
print(t_dat.shape)


#max_v = v_dat = v_dat[-1, :, :].max()
#print(max_v)

#print(dy)
anima_T = animar_dedalus(x, y, T_dat, t_dat, 1./2., 'rainbow_r')
#mywriter = animation.FFMpegWriter()
#anima_T.save('strat_conv_T_180.mp4',writer='imagemagick', fps=40) #nombre de como quieres que se guarde el video. 'imagemagick'

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
