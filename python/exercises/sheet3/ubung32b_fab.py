import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import matplotlib.pyplot as plt
import numpy.random
import math
import scipy.linalg
from scipy.linalg import toeplitz

def mfft(H, unit_sites):

    buff = []
    for i in range(unit_sites):
            buff.append(fft(H[i::unit_sites,:,np.newaxis], axis=0))
    H = np.concatenate(np.concatenate(buff, axis=2), axis=1)
    return H.T

t=-1
t1=0
t2=-1
t22=-0.33j
M=4
N=512
def hamiltion():

#construct unit cell from intra and inter h
	hintra1=np.array([[0,t],[t,0]])
	hintra2=np.array([[0,0],[t1,0]])
	hintray=np.kron(np.eye(M/2),hintra1)
	hintray+=np.kron(np.eye(M/2, k=1),hintra2)
	hintray+=np.kron(np.eye(M/2, k=-1),hintra2.T.conj())
	hintra11=np.array([[0,t1],[t1,0]])
	hintra22=np.array([[0,0],[t,0]])
	hintray1=np.kron(np.eye(M/2),hintra11)
	hintray1+=np.kron(np.eye(M/2, k=1),hintra22)
	hintray1+=np.kron(np.eye(M/2, k=-1),hintra22.T.conj())
	hintrax=np.kron(np.eye(2),[[t2,0],[0,t]])
	M1=np.hstack((hintray,hintrax))
	M2=np.hstack((hintrax,hintray1))
	Hintra=np.vstack((M1,M2))
	#print(Hintra)
	
	#add next nearest hopping
	Hsec=np.array([[0,t22],[-t22,0]])
	Hsec2=np.array([[0,t22],[0,0]])
	Z1=np.hstack((Hsec,Hsec2.T.conj()))
	Z2=np.hstack((Hsec2,Hsec))
	blocksec=np.vstack((Z1,Z2))
	ZZ1=np.hstack((np.zeros((M,M)),blocksec))
	ZZ2=np.hstack((blocksec,np.zeros((M,M))))
	Hsecond=np.vstack((ZZ1,ZZ2))
	Hintra=Hintra+Hsecond
#construct inter hamiltonian 
	hblock=np.zeros(M)
	hblock[::2]=t
	hblock[1::2]=t2
	hinterbb=np.diag(hblock)
	U1=np.hstack((np.zeros((M,M)),hinterbb))
	U2=np.hstack((np.zeros((M,M)),np.zeros((M,M))))
	hinterb=np.vstack((U1,U2))
#add next nearest hopping
	R1=t22*np.eye(M)
	R2=-t22*np.eye(M)
	RR1=np.hstack((R1,np.zeros((M,M))))
	RR2=np.hstack((np.zeros((M,M)),R2))
	hintersec=np.vstack((RR1,RR2))
	hinterb=hinterb+hintersec
#Now construct full hamiltonian 	
	H=np.kron(np.eye(N/2),Hintra)
	H+=np.kron(np.eye(N/2, k=-1), hinterb)
	H+=np.kron(np.eye(N/2, k=1), hinterb.T.conj())
	H+=np.kron(np.eye(N/2, k=-N/2+1), hinterb.T.conj())
	H+=np.kron(np.eye(N/2, k=N/2-1), hinterb)	
	return(H)

def fourierH(H):
	Hfftt=mfft(H, 8)
	Hfftnn=Hfftt.T.conj()
	Hgesamt=mfft(Hfftnn, 8)/N
	#Hgesamt=np.fft.fftshift(Hgesamt)/N
	return(Hgesamt)
	#print(np.dot(H,[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
		
def Eigenvalues(Hgesamt):
	eigenwerte=[]
	eigenwerte2=[]
	for l in range(N/2):
		block=Hgesamt[l*8:8+l*8:,l*8:8+l*8:]
		eigenwerte.append(LA.eigvalsh(block))
		eigenwerte2.extend(LA.eigvalsh(block))
	#print(eigenwerte2)
	return(eigenwerte)
	
def main():
	x=np.linspace(-5,5,N/2)
	y=np.array(Eigenvalues(fourierH(hamiltion())))
	plt.plot(x,y)
	plt.xlabel('k')
	plt.ylabel('E')
	plt.show()

if __name__=="__main__":
    main()
