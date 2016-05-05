import numpy as np
import matplotlib.pyplot as plt

def main():

    x =np.linspace (-5,5,21)
    y =np.linspace (-5,5,21)

    x,y = np.meshgrid(x,y)

    r=np.sqrt(x*x+y*y)

    def vecfield_one(x,y):
        return -y,x
    def vecfield_two(x,y,r):
        return -x/r, -y/r
    def vecfield_three(x,y):
        return -x,-y

    plt.subplot(3,1,1)
    vx, vy =vecfield_one(x,y)
    plt.quiver(x,y,vx,vy,color='Teal')
    
    plt.subplot(3,1,2)
    vx, vy =vecfield_two(x,y,r)
    plt.quiver(x,y,vx,vy,color='Teal')
    
    plt.subplot(3,1,3)
    vx, vy =vecfield_three(x,y)
    plt.quiver(x,y,vx,vy,color='Teal')

    plt.show()


if __name__ == "__main__":
    main()
