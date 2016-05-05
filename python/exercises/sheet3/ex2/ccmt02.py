import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mfft import *
from density_of_states import *

def main():

    L = 200
    M = 4
    t = 1
    t1 = 0
    t2 = t
    t3 = 0.1j*t

    eaxis = np.arange(-5, 5, 0.01)
    eta = 0.1
    z = np.zeros((M, M))

    # construct the Hamiltonian analogously to exercise 3 on sheet 1
    # but add additional terms for the spin-orbit interaction
    intra_ypart1 = -np.diag([t, t1, t], k=1) - np.diag([t, t1, t], k=-1)
    intra_ypart2 = -np.diag([t1, t, t1], k=1) - np.diag([t1, t, t1], k=-1)
    intra_xpart = -np.diag([t2, t, t2, t]) + np.diag([t3, -t3, t3], k=-1) - np.diag([t3, -t3, t3], k=1)
    inter_xpart = -np.diag([t, t2, t, t2]) + np.diag([t3, -t3, t3], k=1)
    inter_ypart = np.diag([t3, -t3, t3, -t3])
    topbuffer =  np.hstack((intra_ypart1, intra_xpart))
    botbuffer = np.hstack((intra_xpart.conj().T, intra_ypart2))
    intra_unitcell_hopping = np.vstack((topbuffer, botbuffer))
    topbuffer = np.hstack((inter_ypart, z))
    botbuffer = np.hstack((inter_xpart, -inter_ypart))
    inter_unitcell_hopping = np.vstack((topbuffer, botbuffer))

    v = np.eye(int(L/2), k=1)
    v[-1,0] = 1

    H_intra = np.kron(np.eye(int(L/2)), intra_unitcell_hopping)
    H_inter = np.kron(v, inter_unitcell_hopping)
    H = H_intra + H_inter + H_inter.conj().T

    #We dont use fftshift here to have the spectrum from 0 to 2pi. The edge states are at k=pi
    Hfourier = mfft(mfft(H,2*M).conj().T,2*M).conj().T/(L/2)

    spectrum = []

    for b in np.arange(0, M*L, 2*M):
        Hblock = Hfourier[b:b+2*M,b:b+2*M]
        energies = np.sort(np.real(LA.eigvals(Hblock)))
        spectrum.append(energies)

    spectrum = np.array(spectrum)
    dens = density_of_states(eaxis, spectrum, eta)

    gauss = dens.gaussian()
    lorentz = dens.lorentzian()

#    #Dispersion relation
    plt.figure(1)
    plt.subplot(111)
    plt.plot(np.linspace(0, 2*np.pi, np.size(spectrum, axis=0)), spectrum)
    plt.suptitle("Dispersion Relation")
    plt.show()

#    #Density of states
    plt.figure(2)
    ax = plt.subplot(111)
    ax.plot(eaxis, gauss)
    ax.plot(eaxis, lorentz)
    plt.suptitle("Density of States")
    plt.show()

    #Brute force the spectrum (to get a sorted list of eigenvalues and eigenstates)
    e, ev = LA.eigh(H)

    #Figure 3: plot a state at the lowest part of the spectrum (first eigenvalue) this should not be an edge state
    xy1 = np.reshape(ev[:,0], (L, M))
    xy2 = np.reshape(ev[:,M*(L/2)-1], (L, M))
    xy_m = np.reshape(ev[:, M*(L/4)-1], (L, M))

    plt.figure(3)
    ax = plt.subplot2grid((2, 1), (0, 0))
    im = ax.imshow((np.absolute(xy1)**2).T, interpolation='none', aspect='auto', cmap=plt.get_cmap('hot'))
    plt.colorbar(im)
#    ax.set_ylim(-5, 5)
    ax.set_title("Low energy state")

    # As comparison we plot a state in the middle of the spectrum
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    im2 = ax2.imshow((np.absolute(xy2)**2).T, interpolation='none', aspect='auto', cmap=plt.get_cmap('hot'))
    plt.colorbar(im2)
    ax2.set_title("State from the middle of the spectrum (edge state)")
    plt.show()

    #Figure 4: real part of the state in Figure 3.1 both at the edge and in the middle part of the sample. In both cases an amplitude is present

    plt.figure(4)
    ax3 = plt.subplot(111)
    line_e, = ax3.plot(np.real(xy1[:,0]))
    line_m, = ax3.plot(np.real(xy1[:,2]))
    ax3.set_title("Real part of low energy state at edge and middle of sample")
    plt.legend([line_e, line_m], ['Edge', 'Middle'])
    plt.show()

    #Figure 5: imaginary part of the state in Figure 3.2 bot at the edge and in the middle part of the sample. Only the edge shows a significant amplitude
    plt.figure(5)
    ax4 = plt.subplot(111)
    line_edge, = ax4.plot(np.imag(xy2[:,0]))
    line_middle, = ax4.plot(np.imag(xy2[:,2]))
    ax4.set_title("Imaginary part of edge state at the edge and the middle of the sample")
    plt.legend([line_edge, line_middle], ['Edge', 'Middle'])
    plt.show()

    #Figure 6: real part of a state somewhere else in the spectrum (a quarter into the eigenvalues), both at the edge and in the middle of the sample
    plt.figure(6)
    ax5 = plt.subplot(111)
    ax5.plot(np.real(xy_m[:,0]))
    ax5.plot(np.real(xy_m[:,(M/2)-1]))

    plt.show()

if __name__ == "__main__":
    main()
