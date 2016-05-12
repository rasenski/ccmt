import numpy as np
import numpy.linalg as LA
from numpy.fft import *
import matplotlib.pyplot as plt
import numpy.random
import math
import scipy.linalg as sLA




L=[30,60,120,240,480]
t=1
epsi=-t/2
ed=-4*t
V=0.1*t

#betrachte als single chain mit spin faktor von 2, matrix nur 1 block// it doesnt matter if you view spin with s=2 or if you increase 1d chain size

def hamiltonian(L,t,epsi,V):
    s=2
    H=np.eye(L, k=0)*0+np.eye(L,k=1)*1+np.eye(L,k=-1)*1
    H[-1,-1]=epsi
    H[-1,-2]=V
    H[-2,-1]=V
    Hs=t*np.kron(np.eye(s), H)
    return(Hs)

def main():
    eta=0.05
    e=np.linspace(-5,5,1000)
    for j in L:
        H=hamiltonian(j,1,epsi,V)
        ew, ev=LA.eigh(H)
        ew=ew.flatten()
        def delta_1(ew,e):
            return (1/j)*np.sum((1/math.pi)*(eta/(eta**2+(e-ew)**2)))
        def delta_2(ew,e):
            return (1/j)*np.sum( (1/(eta*math.sqrt(math.pi)))*np.exp((-(e-ew)**2)/(eta**2)) )
        lorenz_liste=[]
        gauss_liste=[]
        for i in e:
            lorenz_liste.append(delta_1(ew,i))
        for l in e:
            gauss_liste.append(delta_2(ew,l))
        plt.xlabel('DOS')
        plt.ylabel('E')
        plt.subplot(121)
        plt.plot(e,lorenz_liste)
        plt.plot(e,gauss_liste)
    for j in L:
        ew, ev=LA.eigh(hamiltonian(j,1,ed,V))
        ew=ew.flatten()
        d=np.zeros(2*j)
        d[-1]=1
        def delta_1b(ew,e):
            return (1/j)*np.sum(np.abs(np.dot(ev.T,d))**2*(1/math.pi)*(eta/(eta**2+(e-ew)**2)))
        def delta_2b(ew,e):
            return (1/j)*np.sum(np.abs(np.dot(ev.T,d))**2*(1/(eta*math.sqrt(math.pi)))*np.exp((-(e-ew)**2)/(eta**2)) )
        lorenz_liste=[]
        gauss_liste=[]
        for i in e:
            lorenz_liste.append(delta_1b(ew,i))
        for l in e:
            gauss_liste.append(delta_2b(ew,l))
        plt.xlabel('PDOS')
        plt.ylabel('E')
        plt.subplot(122)
        plt.plot(e,lorenz_liste)
        plt.plot(e,gauss_liste)
    plt.show()

        
if __name__== "__main__":
    main()
