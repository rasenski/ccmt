import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import matplotlib.pyplot as plt
import numpy.random
import math
import scipy.linalg
from scipy.linalg import toeplitz


N=120
t=1
V=0.1*t
#ed=-t
ed=[-t,0,t]
Ne=N+1

def alles(ed):


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
	

	eigenwert, eigenvec=LA.eigh(H)
	
	
	
	#construct the density matrix roh 
	roh=np.zeros((2*N+2,2*N+2))
	Imat=np.eye(2*N+2)
	f=np.zeros((2*N+2))
	f[:Ne:]=1
	for j in range(2*N+2):	
		for k in range(2*N+2):
			roh[j,k]=sum(np.dot(Imat[:,j],eigenvec[:,n])*f[n]*np.dot(eigenvec[:,n],Imat[:,k]) for n in range(2*N+2))
	#evaluate the charge and Spin
		
	return(roh[N,N]+roh[-1,-1],(roh[N,N]-roh[-1,-1])/2)

def main():
	x=np.array([-1,0,1])
	#print(alles(ed[1]))
	q=[]
	s=[]
	for j in range(len(ed)):
		q.append(alles(ed[j])[0])
		s.append(alles(ed[j])[1]) 
	plt.subplot(1,2,1)
	plt.plot(x,q,'ro')
	plt.axis([-1.5, 1.5, -0.2, 2.2])
	plt.xlabel('Ed in 1/t')
	plt.ylabel('Charge q')
	plt.subplot(1,2,2)
	plt.plot(x,s,'bo')
	plt.axis([-1.5, 1.5, -1, 1])
	plt.ylabel('Spin Sz')
	plt.xlabel('Ed in 1/t')
	plt.show()
if __name__=="__main__":
    main()
