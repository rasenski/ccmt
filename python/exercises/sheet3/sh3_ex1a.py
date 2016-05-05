import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import matplotlib.pyplot as plt
import numpy.random
import math


k=[1,2,3]

def alles(k):

	N=250
	t=1
	#Define our hamiltonian like before (sheet 1)#
	Hcell=[[0,t],[t,0]]
	Hint=[[0,0],[t,0]]
	#Define an array of random numbers 
	zufallszahlen = np.random.rand(2*N)-0.5
	#Add everything together
	HT = np.transpose(Hint)
	H = np.kron(np.eye(N), Hcell)
	H+=np.kron(np.eye(N, k=1), Hint)
	H+=np.kron(np.eye(N, k=-1), HT)
	H+=np.kron(np.eye(N, k=-N+1), Hint)
	H+=np.kron(np.eye(N, k=N-1), HT)
	H+=np.diag(zufallszahlen)
	
	#Calculate the eigenvalues of the Hamiltonian#
	eigenwerte=LA.eigvals(H)
	
	
	#print(eigenwerte)
	
	eta=0.05
	e=np.linspace(-4,4,2*N)
	
	#print (len(block))#=8*L/2
	#Define the delta-distribution with finite broadening

	def lorenzian(e):
		return sum ((1/(math.pi))*eta/((e-eigenwerte[i])**2+eta**2) for i in range(2*N))/(2*N)
	
	def gaussian(e):
		return sum( 1/(eta*math.sqrt(math.pi))*np.exp(-(e-eigenwerte[i])**2/eta**2) for i in range(2*N))/(2*N)

	plt.plot(e,lorenzian(e), 'r')
	plt.plot(e,gaussian(e))
	plt.ylabel('E')
	plt.xlabel('DOS')
	plt.show() 
	return(eigenwerte)


def main():

	eigenwerte2=[]
	for l in range(len(k)):
		eigenwerte2.append(alles(k[l]))

	plt.plot(k, eigenwerte2,'r.', markersize=0.5)
	plt.axis([0, 4, -3, 3])
	plt.ylabel('E')
	plt.show()
	


if __name__=="__main__":
    main()
