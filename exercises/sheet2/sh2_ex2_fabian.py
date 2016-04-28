import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import matplotlib.pyplot as plt
import math
import cmath
from math import e 

def mfft(H, unit_sites):

    buff = []
    for i in range(unit_sites):
            buff.append(fft(H[i::unit_sites,:,np.newaxis], axis=0))
    H = np.concatenate(np.concatenate(buff, axis=2), axis=1)
    return H.T
phi=[1.0/42, 1.0/21, 1.0/7, 4.0/21, 8.0/21, 2.0/7, 1.0/2]
def main():

#construct intra hamiltionian 
	L=42
	t=1
	p=phi[0]
	Hcell=[[0,t],[t,0]]
	Hint=[[0,0],[t,0]]
	HT = np.transpose(Hint)
	Hintray1 = np.kron(np.eye(L/2), Hcell)+np.kron(np.eye(L/2, k=1), Hint)+np.kron(np.eye(L/2, k=-1), HT)
	Hintray1+=np.kron(np.eye(L/2, k=-L/2+1), Hint)
	Hintray1+=np.kron(np.eye(L/2, k=L/2-1), HT)
	Hintray=np.kron(np.eye(2),Hintray1)
	
	a=[]
	for i in range(L):
		a.append(e**(-1j*p*i*2*math.pi))
	
	intrax=np.diag(a)
	
	Hin=np.kron([[0,1],[0,0]], intrax)
	Hin+=np.kron([[0,0],[1,0]], np.transpose(np.conjugate(intrax)))
	Hin+=Hintray	
	
	Hintray1=np.kron([[0,0],[1,0]], intrax)
	HintrayT=np.transpose(np.conjugate(Hintray1))
	
	H=np.kron(np.eye(L/2),Hin)+np.kron(np.eye(L/2, k=1),Hintray1)+np.kron(np.eye(L/2, k=-1),HintrayT)
	H+=np.kron(np.eye(L, k=-L+1), intrax)
	H+=np.kron(np.eye(L, k=L-1), np.conjugate(intrax))
	

	eigenwerte=LA.eigvals(H)

	x=np.linspace(-5, 5, L*L, endpoint=True)
	plt.plot(x, eigenwerte, 'ro')
	plt.show()


	
	
	
	

if __name__=="__main__":
    main()
