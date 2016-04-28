import numpy as np
import numpy.linalg as LA
from numpy.fft import *
import matplotlib.pyplot as plt
import numpy.random
import math


def disorder(k):
#number of sites
    L=500
#hopping matrix t_kl
    t=1
#hamiltonian is defined by
# H=-sum_kl t_kl c^t_k*c_l + sum_l\inL eps_l*c^t_lc_l
# which is the same as in sh1_ex1. hc=hcell, hi=hinter
    hc=np.array([[0,t],[t,0]])
    hi=np.array([[0,0],[t,0]])
#define array with rnd numers. np.random.rand(d0,d1,..,dn) generates a rnd number list (if not given an additional argument) from [0,1)#
    rnd=np.random.rand(2*L)-0.5
#construct hamiltonian 
#start with 1d tight binding model
    H=-np.kron(np.eye(L), hc)
    H+=np.kron(np.eye(L, k=1), hi)
    H+=np.kron(np.eye(L, k=-1), hi.T)
#add boundary conditions
    H+=np.kron(np.eye(L, k=-L+1), hi)
    H+=np.kron(np.eye(L, k=L-1), hi.T)
#add random diagonal term
    H+=np.diag(rnd)
#calc eigenvals of hamiltonian
    ev=LA.eigvalsh(H)
    print(ev)
#pseudocode: definiere calc_dos(e,ev,L) die mir dos returned
# dann np.avarage? über 1000x calc_dos. plotte nun 


#calculate DOS
    k=np.linspace(-np.pi,np.pi,L)
    e=np.linspace(-5,5,L)
    eta=0.05
    def delta_1(ev,e):
        return (1/L)*np.sum((1/math.pi)*(eta/(eta**2+(e-ev)**2)))
    def delta_2(ev,e):
        return (1/L)*np.sum( (1/(eta*math.sqrt(math.pi)))*np.exp((-(e-ev)**2)/(eta**2)) )
    lorenz_liste=[]
    gauss_liste=[]
    for i in e:
        lorenz_liste.append(delta_1(ev,i))
    for l in e:
        gauss_liste.append(delta_2(ev,l))
#plot DOS
    plt.xlabel('DOS')
    plt.ylabel('E')
    plt.plot(lorenz_liste,e)
    plt.plot(gauss_liste,e)
    plt.show()
    return(ev)


def main():
    lst=[1,2,3]
    for i in lst:
        disorder(i)

if __name__=="__main__":
    main()

