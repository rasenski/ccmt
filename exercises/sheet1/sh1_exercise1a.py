import numpy as np
from numpy.fft import *
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mfft(K, unit_sites):
    foo=[]
    for i in range(unit_sites):
        foo.append(fft(K[i::unit_sites,:,np.newaxis], axis=0))
    K=np.concatenate(np.concatenate(foo, axis=2), axis=1)
    return K.T

def main():
    N=512
    
    t=1
    dt=1
    j=t+dt

    intra=np.array([[0,t],[t,0]])
    inter=np.array([[0,0],[j,0]])
    inter_T=inter.T

    H=np.kron(np.eye(N), intra)
    H +=np.kron(np.eye(N, k=1), inter)
    H +=np.kron(np.eye(N, k=-1), inter_T)
    H +=np.kron(np.eye(N, k=N-1), inter_T)
    H +=np.kron(np.eye(N, k=-N+1), inter)

    print(inter)
    print(intra)
    print(H)
    
    Hmfft=mfft(H,2)
    Hmfft2=mfft(Hmfft.conj().T,2).conj().T
    Hmfft3=fftshift(Hmfft2)/N
    print(Hmfft3)

    eigenwerte=[]
    for l in range(N):
        block0=Hmfft3[l*2:2+l*2:,l*2:2+l*2:]
        eigenwerte.extend(LA.eigvals(block0))
    print(eigenwerte)
    
    x=np.arange(N*2)
    y=np.array(eigenwerte)
    plt.plot(x,y, 'ro')
    plt.show()

if __name__ == "__main__":
    main()

