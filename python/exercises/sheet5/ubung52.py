import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import matplotlib.pyplot as plt
import numpy.random
import math
import scipy.linalg
from scipy.linalg import toeplitz


N=30
t=1
V=0.1*t
#ed=-t
ed=[-t,0,t]
beta=[10,1,0.1,0.01]
Ne=N+1
def alles(ed,beta):


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
	
	
	def U(mu):
		return sum(1/(1+np.exp(beta*(eigenwert[i]-mu))) for i in range(2*N+2))-Ne
	
#		def Ulow(mu):
#			kT=0.001
#			Ne=N+1
#			eigenwert=LA.eigvalsh(ham(N,t,V,ed))
#			return sum(np.exp(-(eigenwert[i]-mu)/kT) for i in range(2*N+2))-Ne
	def bisection(a,b,tolerance):
		c = (a+b)/2.0
		while abs(U(c)) > tolerance:
			if U(c) == 0:
				return c
			elif U(a)*U(c) < 0:
				b = c
			else :
				a = c
			c = (a+b)/2.0
		
		return c
	#construct the density matrix roh 
	roh=np.zeros((2*N+2,2*N+2))
	mu1=bisection(1,-1,0.0001)
	Imat=np.eye(2*N+2)
	for j in range(2*N+2):	
		for k in range(2*N+2):
			roh[j,k]=sum(np.dot(Imat[:,j],eigenvec[:,n])*1/(1+np.exp(beta*(eigenwert[n]-mu1)))*np.dot(eigenvec[:,n],Imat[:,k]) for n in range(2*N+2))
		#evaluate the charge
		
	return(roh[N,N]+roh[-1,-1])

def main():
	x=np.array([-1,0,1])
	q=[]
	for k in beta:	
		for j in range(len(ed)):
			q.append(alles(ed[j],k)) 
		plt.subplot(4,1,(beta.index(k)+1))
		plt.plot(x,q[3*beta.index(k):3+3*beta.index(k)],'ro', label="q for beta="+str(k))
		plt.axis([-1.2, 1.2, -0.05, 2.1])
		#plt.xlabel('Ed in 1/t')
		plt.ylabel('Charge q')
		plt.legend(loc=0)
	plt.show()
	

if __name__=="__main__":
    main()
