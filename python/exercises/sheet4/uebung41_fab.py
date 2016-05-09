import numpy as np
import numpy.linalg as LA
from numpy.fft import fft
import matplotlib.pyplot as plt
import numpy.random
import math
import scipy.linalg
from scipy.linalg import toeplitz


#Define our main variables
N=250 #500/2
t=1
M=20
#We define our hamiltonian here since we want the same hamiltonian to compare our results
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
H=-H
H+=np.diag(zufallszahlen)

Ha=np.zeros((M,M))
#Define the vector as given on the sheet
def vectorv():
	v=np.ones(2*N)
	return(v)



#Define our basis-set of non-orthonormal vectors, remind yourself that the rows are our vectors
def basis(H, v):

	basisset=[]
	for l in range(M):
		basisset.extend(np.dot(LA.matrix_power(H,l),v))
	basisset=np.reshape(basisset,(M,2*N))
	return (basisset)



#define classical gram-schmidt orthogonalization
def gsclas(u, v1): 
	w=np.zeros((M,2*N))
	for l in range(M-1):
		w[l+1]=u[l+1]
		for k in range(l+1):
			w[l+1]=w[l+1]-np.dot(v1[k],u[l+1])*v1[k]
		v1[l+1]=w[l+1]/LA.norm(w[l+1]) 
#the rows are our new basisset
	return(v1)



#define the modified gram-schmidt orthogonalization
def gsmod(u,v3):
	w1=np.zeros((M,2*N))
	for l in range(M-1):
		w1[l+1]=u[l+1]
		for k in range(l+1):
			w1[l+1]=w1[l+1]-np.dot(v3[k],w1[l+1])*v3[k]
		v3[l+1]=w1[l+1]/LA.norm(w1[l+1]) 
#the rows are our new basisset
	return(v3)



#define a function with diagonlize the given hamiltonian Ha for a giben basis
def diagonalization(p):

	for i in range(M):
		for j in range(M):
			Ha[i,j]=np.dot(p[i],np.dot(H,p[j]))
	eigenvals=LA.eigvals(Ha)
	return(Ha,eigenvals)


#Arnoli method	
def arnoli(H):
	h=np.zeros((M+1,M))
	v2=np.zeros((M+1,2*N))
	v2[0]=vectorv()/LA.norm(vectorv())
	w3=np.zeros((M,2*N))
	A=H
	for j in range(M):
		for i in range(j+1):
			h[i,j]=np.dot(v2[i],np.dot(A,v2[j]))
		w3[j]=np.dot(A,v2[j])-sum(h[k,j]*v2[k] for k in range(j+1))
		h[j+1,j]=LA.norm(w3[j])
		if LA.norm(w3[j])==0:#include break condition
			break
		v2[j+1]=w3[j]/h[j+1,j]
		hquer=h[:-1:]
	return(v2,h,hquer)

def test(z):

	q=np.zeros((M,M))
	for i in range(M):
		for j in range(M):
			q[i,j]=np.dot(z[i],z[j])
	return(q)


def main():


	#here we define our basissets from the Krylov subspace 
	u=basis(H,vectorv()) #u-vectors
	v=u #v-vectors
	w=u #w-vectors
	v[0]=vectorv()/LA.norm(vectorv()) #v-vector1
	
	
	x=np.arange(M)
	y=np.arange(2*N)

	 
	#plot the test for orthogonalization
	plt.subplot(1,3,1)
	plt.imshow(test(gsclas(u,v)),\
           interpolation='none',\
           aspect='auto')
	plt.colorbar()
	plt.title("Classical")
	plt.subplot(1,3,2)
	plt.imshow(test(gsmod(u,v)),\
           interpolation='none',\
           aspect='auto')
	plt.colorbar()

	plt.title("Modified")

	
	plt.subplot(1,3,3)
	plt.imshow(test(arnoli(H)[0]),\
           interpolation='none',\
           aspect='auto')
	plt.colorbar()

	plt.title("Arnoli")
	plt.show()

	#plot a density plot of the entries for Hc
	plt.imshow(arnoli(H)[1],\
           interpolation='none',\
           aspect='auto')
	plt.colorbar()

	plt.title("Hc")
	plt.show()
	
#	print the eigenvalues for gsclas, gsmod and arnoldi
	plt.subplot(1,4,1)
	plt.plot(x,diagonalization(gsclas(u, v))[1],'ro')
	plt.subplot(1,4,2)
	plt.plot(x,diagonalization(gsmod(u,v))[1],'bo')
	plt.subplot(1,4,3)
	plt.plot(x,LA.eigvals(arnoli(H)[2]),'go')
	plt.subplot(1,4,4)
#	print the eigenvalues for H directly
	plt.plot(y,LA.eigvals(H), 'y.')
	plt.show()

	print(np.dot(gsclas(basisset3, basisset1, basisset2)[0],gsclas(basisset3, basisset1, basisset2)[1]))
	print(np.dot(gsmod(basisset3, basisset2)[0],gsmod(basisset3, basisset2)[1]))
	print(np.dot(arnoli(H)[0][0],arnoli(H)[0][1]))

	
	


if __name__=="__main__":
    main()
