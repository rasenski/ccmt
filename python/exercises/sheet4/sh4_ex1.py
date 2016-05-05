import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import  matplotlib.pyplot as plt
import numpy.random, math, scipy.linalg
from scipy.linalg import toeplitz


L=100
t=1
M=25

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


def vec():
    v=np.ones(2*L)
    return(v)

def base(H,v):
    basis=[]
    for i in range(M):
        basis.append(np.dot(LA.matrix_power(H,i),v.T))
    return(basis)

#def base(H,v):
#    basis=[]
#    for i in range(M):
#        basis.extend(np.dot(LA.matrix_power(H,i),v))
#    basis=np.reshape(basis,(M,2*L))
#    return(basis)

def gscl(u, v):
    w=np.zeros((M,2*L))
    for j in range(M-1):
        w[j+1]=u[j+1]
        for k in range(j+1):
            w[j+1]=w[j+1]-np.dot(v[k].T,u[j+1])*v[k]
        v[j+1]=w[j+1]/LA.norm(w[j+1])
    return(v)


def gsm(u, v):
    w1=np.zeros((M,2*L))
    for j in range(M-1):
        w1[j+1]=u[j+1]
        for k in range(j+1):
            w1[j+1]=w1[j+1]-np.dot(v[k].T,w1[j+1])*v[k]
        v[j+1]=w1[j+1]/LA.norm(w1[j+1])
    return(v)


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
    plt.subplot(1,3,1)
    plt.imshow(foo(gscl(u,v)), interpolation='none', aspect='auto')
    plt.colorbar()
    plt.title("classical gram schmidt")
    plt.subplot(1,3,2)
    plt.colorbar()
    plt.title("modified gram schmidt")
    plt.imshow(foo(gsm(u,v)), interpolation='none', aspect='auto')
    plt.subplot(1,3,3)
    plt.imshow(foo(arnoldi(H)[0]), interpolation='none', aspect='auto')
    plt.colorbar()
    plt.title("Arnoldi Method")
    plt.show()

if __name__=="__main__":
    main()
