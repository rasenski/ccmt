import numpy as np
import numpy.linalg as LA
from numpy.fft import *
import matplotlib.pyplot as plt
import math

def mfft(U, unit_sites):
    buff =[]
    for i in range(unit_sites):
        buff.append(fft(U[i::unit_sites,:,np.newaxis], axis=0))
    U = np.concatenate(np.concatenate(buff, axis=2), axis=1)
    return U.T

def main():
    for i in range(0,10,1):
        i=i/10
        blubb(1,i,1-i)

def blubb(t,t1,t2):
    N=256
    
    #construct unit cell from intra and inter h
    hintra1=np.array([[0,t],[t,0]])
    hintra2=np.array([[0,t2],[t2,0]])
    hinter1=np.array([[t,0],[0,t1]])
    hinter2=hinter1.T
    #stack little hs to get unit cell intras
    M1=np.hstack((hintra1,hinter1))
    M2=np.hstack((hinter2,hintra2))
    M=np.vstack((M1,M2))
    heinheitintra=np.kron(np.eye(2), M)
    #build stack for inters for 8x8 unit cell
    hinter3=np.array([[t1,0],[0,t]])
    Z=np.zeros((2,2))
    K1=np.hstack((Z,Z))
    K2=np.hstack((hinter3,Z))
    K=np.vstack((K1,K2))
    K_T=K.T
    heinheitinter1=np.kron(np.eye(2,k=+1), K)
    heinheitinter2=np.kron(np.eye(2,k=-1), K_T)
    #put together to get h einheit
    heinheit=heinheitintra+heinheitinter1+heinheitinter2
#    print(heinheit)
    #construct inter for chi the big hamiltonian ribbon
    c1=np.array([[0,0],[t2,0]])
    c2=np.array([[0,0],[t,0]])
    chi1=np.hstack((c1,Z))
    chi2=np.hstack((Z,c2))
    chi=np.vstack((chi1,chi2))
    Z2=np.zeros((4,4))
    heinheitinter=np.vstack((np.hstack((chi,Z2)),np.hstack((Z2,chi))))
#    print(heinheitinter)
    #now put it all together + boundary conditions
    H=np.kron(np.eye(N), heinheit)
    H+=np.kron(np.eye(N,k=1),heinheitinter.T)
    H+=np.kron(np.eye(N,k=-1),heinheitinter)
    H+=np.kron(np.eye(N,k=N-1), heinheitinter.T)
    H+=np.kron(np.eye(N,k=-N+1), heinheitinter)
#    print(H)
    
    hmfft=(1/N)*fftshift(mfft(mfft(H.conj().T,8).conj().T,8))
#    print(hmfft)

    eigenwerte=[]
    for l in range(N):
            block=hmfft[l*8:8+l*8:,l*8:8+l*8:]
            eigenwerte.append(LA.eigvalsh(block))
#    print(eigenwerte)
    k=np.linspace(-5,5,N)
    y=np.array(eigenwerte)

#   define dos
    k2=np.linspace(-5,5,8*N)
    e=np.linspace(-3,3,8*N)
    eta=0.05
    ew=y.flatten()
    def delta_1(ew,e):
        return (1/N)*np.sum((1/math.pi)*(eta/(eta**2+(e-ew)**2)))
    def delta_2(ew,e):
        return (1/N)*np.sum( (1/(eta*math.sqrt(math.pi)))*np.exp((-(e-ew)**2)/(eta**2)) )
    lorenz_liste=[]
    gauss_liste=[]
    for i in e:
        lorenz_liste.append(delta_1(ew,i))
    for l in e:
        gauss_liste.append(delta_2(ew,l))
    
    plt.subplot(1,2,1)
    plt.plot(k,y)
    plt.xlabel('k')
    plt.ylabel('E')
    plt.subplot(1,2,2)
    plt.plot(lorenz_liste,e)
    plt.plot(gauss_liste,e)
    plt.show()

if __name__ == "__main__":
    main()
