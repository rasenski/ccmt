import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import matplotlib.pyplot as plt
import numpy.random
import math
import scipy.linalg
from scipy.linalg import toeplitz


N=[30,60,120,240,480]
t=1
V=0.1*t
ed=-4*t

def ham(N,t,V,ed):

#define the lead hamiltonian
	Hl =t*np.eye(N, k=1)	
	Hl+=t*np.eye(N, k=-1)
	#Hl[0,-1]=1
	#Hl[-1,0]=1
	#define dot hamiltonian
	Hd=ed
	#define interaction hamiltonian
	Hld=V
	row=np.zeros(N)
	row[-1]=V
	col=np.zeros((N+1,1))
	col[-2]=V
	col[-1]=ed
	H=np.vstack((Hl,row))
	H=np.hstack((H,col))
	#include spin
	H=np.kron(np.eye(2),H)
	return(H)

def main():
	
	for j in N:
		eigenwerte, eigenvec=LA.eigh(ham(j,t,V,-t/2))
		eta=0.05
		e=np.linspace(-4,4,1000)
		def lorenzian(e):
			return sum ((1/(math.pi))*eta/((e-eigenwerte[i])**2+eta**2) for i in range(2*j+2))/(2*j+2)
	
		def gaussian(e):
			return sum( 1/(eta*math.sqrt(math.pi))*np.exp(-(e-eigenwerte[i])**2/eta**2) for i in range(2*j+2))/(2*j+2)
		
		plt.plot(e,lorenzian(e), 'r')
		plt.plot(e,gaussian(e))
		plt.ylabel('E')
		plt.xlabel('DOS')
		plt.show()
	
	for j in N:
		eigenwerte, eigenvec=LA.eigh(ham(j,t,V,10*t))
		eta=0.10
		e=np.linspace(-5,11,1000)
		d=np.zeros(2*j+2)
		d[-1]=1
		def lorenzian2(e):
			return sum (abs(np.dot(eigenvec[:,i],d))**2*(1/(math.pi))*eta/((e-eigenwerte[i])**2+eta**2) for i in range(2*j+2))
	
		def gaussian2(e):
			return sum(abs(np.dot(eigenvec[:,i],d))**2*1/(eta*math.sqrt(math.pi))*np.exp(-(e-eigenwerte[i])**2/eta**2) for i in range(2*j+2))
		
		plt.plot(e,lorenzian2(e), 'r')
		plt.plot(e,gaussian2(e))
		plt.ylabel('E')
		plt.xlabel('A')
		plt.show()
	
if __name__=="__main__":
    main()
