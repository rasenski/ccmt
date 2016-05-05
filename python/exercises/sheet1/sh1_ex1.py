import numpy as np
import numpy.linalg as LA
from numpy.fft import fft

def mfft(H, unit_sites):

    buff = []
    for i in range(unit_sites):
            buff.append(fft(H[i::unit_sites,:,np.newaxis], axis=0))
    H = np.concatenate(np.concatenate(buff, axis=2), axis=1)
    return H.T

def main():
    N=512
    Hn=np.eye(N,k=-1)
    Hp=np.eye(N,k=+1)
    
    dt1 = 1
    dt2 = 0.1
    dt3 = 0.01
    dt4 = 0

    t=1+dt1

    Hn[::2]*=t
    Hp[1::2]*=t
    
    H=Hn+Hp
    
#    eig=LA.eigvals(H)
#    Hdiag = np.full((N,N),0)
#    np.fill_diagonal(Hdiag, eig)
#    print(Hdiag)
#    Hfft= mfft(H, 2)
#    HfftT = Hfft.T
#    print (HfftT)
#    HfftTC=np.conjugate(HfftT)
#    print(HfftTC)
#    Hrichtig = mfft(HfftTC,2)
#    print (Hrichtig)
    print(H)
    Hfft= mfft(H, 2)
    Hfft2=mfft(Hfft.conj().T,2)
    np.set_printoptions(precision=2)
    print(Hfft2)
if __name__=="__main__":
    main()
