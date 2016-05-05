import numpy as np
from numpy.fft import fft

def mfft(H, unit_sites):

    buff = []
    for i in range(unit_sites):
    ##This function splits the Hamiltonian into the contributions of the sites inside the unit cell
    ##The fourier transform is evaluated for each of those sites seperately
    ##The newaxis is introduced to glue the fourier transformed part of the sites back together.
        buff.append(fft(H[i::unit_sites,:,np.newaxis], axis=0))     #Split array into the sites of the unit cell

    H = np.concatenate(np.concatenate(buff, axis=2), axis=1)        #Put seperate fourier transforms back together

    return H.T
