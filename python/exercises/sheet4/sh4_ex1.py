import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import  matplotlib.pyplot as plt
import numpy.random, math, scipy.linalg
from scipy.linalg import toeplitz

#parameters L=sites, t=hamiltonian entry, M=size of krylov space
L=500
t=1
M=50
#hamiltonian is defined by
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
#initialize an empty hamiltonian for diagonalization
Hd=np.zeros((M,M))
#initialize a vector with elements 1 
def vec():
    v=np.ones(2*L)
    return(v)
#construct the base set of non orthonormal vectors (rows=vectors)
def base(H,v):
    basis=[]
    for i in range(M):
        basis.append(np.dot(LA.matrix_power(H,i),v.T))
    return(basis)
#diagonalization function for given hamiltonian
def diago(p):
    for i in range(M):
        for j in range(M):
            Hd[i,j]=np.dot(p[i],np.dot(H,p[j]))
    ev=LA.eigvals(Hd)
    return(Hd,ev)
#classical gram schmidt orthogonalization, takes base set and non-orthogonal vector, returns orthogonal vector 
def gscl(u, v):
    w=np.zeros((M,2*L))
    for j in range(M-1):
        w[j+1]=u[j+1]
        for k in range(j+1):
            w[j+1]=w[j+1]-np.dot(v[k].T,u[j+1])*v[k]
        v[j+1]=w[j+1]/LA.norm(w[j+1])
    return(v)

#modified gram schmidt ..
def gsm(u, v):
    w1=np.zeros((M,2*L))
    for j in range(M-1):
        w1[j+1]=u[j+1]
        for k in range(j+1):
            w1[j+1]=w1[j+1]-np.dot(v[k].T,w1[j+1])*v[k]
        v[j+1]=w1[j+1]/LA.norm(w1[j+1])
    return(v)

#arnoldi mathod takes given matrix (in our case sh3_ex1) and a vector, returns upper hessenberg matrix. get spectrum by eliminating last row
def arnoldi(H):
	h=np.zeros((M+1,M))
	v=np.zeros((M+1,2*L))
	v[0]=vec()/LA.norm(vec())
	w=np.zeros((M,2*L))
	for j in range(M):
		for i in range(j+1):
			h[i,j]=np.dot(v[i],np.dot(H,v[j]))
		w[j]=np.dot(H,v[j])-sum(h[k,j]*v[k] for k in range(j+1))
		h[j+1,j]=LA.norm(w[j])
		if LA.norm(w[j])==0:#break condition
			break
		v[j+1]=w[j]/h[j+1,j]
		hquer=h[:-1:]
	return(v,h,hquer)

#test function to apply orthogonalizations on
def foo(z):
    q=np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            q[i,j]=np.dot(z[i],z[j])
    return(q)


def main():
    u=base(H,vec())
    v=u
    v[0]=vec()/LA.norm(vec())

#plot the hamiltonians as density plots with imshow to analize numerical errors
    plt.subplot(1,4,1)
    plt.imshow(foo(gscl(u,v)), interpolation='none', aspect='auto')
    plt.colorbar()
    plt.title("classical gram schmidt")
    plt.subplot(1,4,2)
    plt.colorbar()
    plt.title("modified gram schmidt")
    plt.imshow(foo(gsm(u,v)), interpolation='none', aspect='auto')
    plt.subplot(1,4,3)
    plt.imshow(foo(arnoldi(H)[0]), interpolation='none', aspect='auto')
    plt.colorbar()
#plot the hamiltionian of the arnoldi method
    plt.title("Arnoldi Method")
    plt.subplot(1,4,4)
    plt.imshow(arnoldi(H)[1], interpolation='none', aspect='auto')
    plt.colorbar()
    plt.title("Density plot for Hamiltonian")
    plt.show()
#plot different eigenvalue spectra for the different methods and compare in one plot
    x=np.arange(M)
    y=np.arange(2*L)
    plt.subplot(141)
    plt.plot(x,diago(gscl(u,v))[1], 'ro', markersize=2)
    plt.subplot(142)
    plt.plot(x,diago(gsm(u,v))[1], 'bo', markersize=2)
    plt.subplot(143)
    plt.plot(x,LA.eigvals(arnoldi(H)[2]), 'go', markersize=2)
    plt.subplot(144)
    plt.plot(y,LA.eigvals(H), 'yo',markersize=2)
    plt.show()


if __name__=="__main__":
    main()
