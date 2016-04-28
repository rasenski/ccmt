import numpy as np
from numpy.fft import fft

def mfft(H, unit_sites):

    buff = []
    for i in range(unit_sites):
    ##This function splits the Hamiltonian into the contributions of the sites inside the unit cell
    ##The fourier transform is evaluated for each of those sites seperately
    ##The newaxis is introduced to glue the fourier transformed Hamiltonians back together.
        buff.append(fft(H[i::unit_sites,:,np.newaxis], axis=0))     #Split array into the sites of the unit cell

    H = np.concatenate(np.concatenate(buff, axis=2), axis=1)        #Put seperate fourier transforms back together

    return H.T


def main():
    N = 92

    X = np.array([[0,1j],[-1j,0]])
    Y = np.array([[2j,0],[0,-2j]])

    M_X = np.kron(np.eye(4), X)
    M_Y = np.kron(np.eye(4), Y)
    M_Y_T = M_Y.T.conj()

    Hneu = np.kron(np.eye(N), M_X)
    Hneu += np.kron(np.eye(N, k=1), M_Y)
    Hneu += np.kron(np.eye(N, k=-1), M_Y_T)
    
    print(Hneu)
    Hmfft=mfft(Hneu,16)
    Hmfft2=mfft(Hmfft.conj().T,16)
    np.set_printoptions(precision=2)
    (Hmfft2)

if __name__ == "__main__":
    main()
