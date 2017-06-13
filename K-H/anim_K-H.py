import h5py
from matplotlib import animation

#Función para extraer datos del archivo hdf5

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
