import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from matplotlib import animation
import matplotlib.colors as colors

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
    fig, axis = plt.subplots(figsize=(4,7))
    p = axis.pcolormesh(xm, ym, S[0,:,:],  norm= colors.PowerNorm(gamma=norma), cmap=CMAP)
    plt.colorbar(p)

    def init():
                print('update init')
                p.set_array(np.ravel(S[0,:-1,:-1]))

                return p

    def update(frame):
        p.set_array(np.ravel(S[frame, :-1, :-1]))
        plt.title(str(t[frame]))
        return p

    anim = animation.FuncAnimation(fig, update, frames= [i for i in range(0,len(S))], init_func=init,  blit = False)
    plt.show()
    return anim

#Abajo tienes que poner el nombre del archivo hdf5 en donde guardaste los datos.

T_dat , ρ_dat, t_dat = extraer_datos('strat_conv_analysis/strat_conv_analysis.h5')
print('sim t')
print(t_dat.shape)


#max_v = v_dat = v_dat[-1, :, :].max()
#print(max_v)

#print(dy)
anima_T = animar_dedalus(x, y, T_dat, t_dat, 1./2., 'rainbow_r')
mywriter = animation.FFMpegWriter()
anima_T.save('strat_conv_T.mp4',writer=mywriter, fps=50) #nombre de como quieres que se guarde el video.

'''
anima_ρ = animar_dedalus(x, y, ρ_dat, 'rainbow')
#mywriter = animation.FFMpegWriter()
anima_ρ.save('RB_conv_rho.mp4',writer=mywriter, fps=30) #nombre de como quieres que se guarde el video.


print(T_dat.shape)
#
fig, axis = plt.subplots(figsize=(4,7))
pT = axis.pcolormesh(x, y, T_dat[-1,:,:], norm= colors.PowerNorm(gamma=1./2.), cmap='rainbow');
#pT = axis.pcolormesh(x, y, T_dat[-1,:,:], vmin=0, vmax=4.3, cmap='rainbow');

plt.colorbar(pT)
plt.title('Temperature, frame 240')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''
