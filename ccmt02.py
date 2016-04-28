import numpy as np
import numpy.linalg as LA
import scipy.linalg as sLA
import matplotlib.pyplot as plt

def main():

    L = 42
    t = 1
    h_0 = t*np.eye(L, k=1)                                      ##Upper triagonal of hopping in y-direction
    h_0[0,L-1] = np.conjugate(t)                                ##Periodic boundary condition in -y-direction
    h_0 = h_0 + h_0.conj().T                                    ##Full matrix for one slice in y
    h_0 = np.kron(np.eye(L), h_0)                               ##Put all slices together

    diag = np.eye(L, k=1)
    period = np.eye(L, k=L-1)

##Now only the hopping between 1d slices is missing the for loop will add these

#    ps = np.array([1, 2, 6, 8, 12, 14, 16, 21])
    ps = np.linspace(0, 100, 50)                               ##All fluxes
    spectrum = []

    for p in np.nditer(ps):
        t_mag = np.exp(2j*np.pi*np.arange(L)*p/L)               ##Peierls phase for all fluxes
        X = t*np.diag(t_mag)
        X = np.kron(diag, X) + np.kron(period, X.conj().T)
        X = X + X.conj().T                                      ##Construct the hopping matrix in x-direction between 1d slices
        H = h_0 + X                                             ##Add the two hamiltonian parts
        spectrum.append(LA.eigvalsh(H))                         ##Get eigenvalues for each flux

    ax = plt.subplot(111)
    ps = ps/L
    ax.plot(ps, spectrum, marker='.', markersize=0.1, linestyle='None', color='k')

    plt.show()



if __name__ == "__main__":
    main()
