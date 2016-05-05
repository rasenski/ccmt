import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():

    x=np.linspace(-5, 5, 100)
    y=np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x,y)
    r = x*x + y*y

    def func_one(x,y):
        return x*y
    def func_two(r):
        return 1/(1+np.power(r, 3))
    def func_three(r):
        return np.exp(-r)
#    plt.subplot(3, 1, 1)
#    plt.pcolor(x,y, func_one(x,y), cmap=plt.get_cmap('cool'))
#    plt.colorbar()

#    plt.subplot(3,1,2)
#    plt.pcolor(x,y, func_two(r), cmap=plt.get_cmap('cool'))

#    plt.subplot(3,1,3)
#    plt.pcolor(x,y, func_three(r), cmap=plt.get_cmap('cool'))

#    plt.show()


#    plt.subplot(3,1,1)
#    CS = plt.contour(x,y,func_one(x,y))
#    plt.clabel(CS, inline=1, fontsize=10)
#    plt.show()

    fig =plt.figure()
    ax = fig.add_subplot(1,1,1, projection ='3d')
    ax.plot_surface(x,y, func_one(x,y))

    plt.show()

if __name__ == "__main__":
    main()
