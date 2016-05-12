import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


#The Hamilton will be constructed out of the Hamiltonian of the leads, the hamiltonian of
#the dot and the hamiltonian, that couples the dot with the lead

#The hamiltonian of the leads is the hamiltonan of a thight binding chain
#The additional spin degree of freedom is implemented by replacing the coupling constant t
#with a 2x2 unit matrix mulitplied with t

#The hamiltonian of the dot is simply a unitary matrix, since the spin up/down particle
#only couples with itself

#The coupling of the leads with the dot is basically like the norm tight binding chain,
#but with a different coupling constant

def hamiltonian(t, v, e, L):
    #hamiltonian of the leads
    h_leads = t*np.kron(np.eye(L, k=1) + np.eye(L, k=-1), np.eye(2))
#    print(h_leads, '\n')

    #hamiltonian of the dot
    h_dot = e*np.eye(2)
#    print(h_dot, '\n')

    #hamiltonian, that couples the dot with the chain
    h_couple = np.zeros((L))
    h_couple[L-1] = v
    h_couple = np.kron(h_couple, np.eye(2))
#    print(h_couple.T, '\n')

    #Stacking of the hamiltonian
    hamiltonian = np.vstack((np.hstack((h_leads, h_couple.T)), np.hstack((h_couple, h_dot))))
#    print(hamiltonian, '\n')

    return hamiltonian

#Gaussian delta-function to compute the DOS
def gaussian(eta, epsi):
    return np.exp(-epsi**2/eta**2)/(eta*np.sqrt(np.pi))
    
    

    
    
#Mainfunction
def main():

    #Parameters for the whloe exercise
    t = 1
    v = 0.1*t
    eta = 0.02

    #Parameters for assignment 1a

    e = -0.5*t
    L = [30, 60, 120, 240, 480]

    #hamiltonian for L = 30, 60, 120, 240, 480
    ham_val = []
    x = []
    for i in L:
        ham = hamiltonian(t, v, e, i)
        #computing of the eigenvalues of the hamiltonian
        ham_val += [la.eigvalsh(ham)]
    x = np.linspace(-5, 5, 2000)

    #plotting of the DOS
    plt.figure(1)
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.plot(x, 1/L[i]*sum([gaussian(eta, x-epsi_k) for epsi_k in ham_val[i]]) , label="Density of states for" + str(L[i]) + "Leads")
        plt.legend(loc = 0)

    
    #Parameters for assignment 1b
    e = -4*t
    L = 480

    #Computation of Eigenvalues/states of the hamiltonian
    ham = hamiltonian(t, v, e, L)
    values, states = la.eigh(ham)

    #Dot Orbital for Spin up/down
    orb_up = np.eye(1, M=2*L+2, k=2*L)[0]
    orb_down = np.eye(1, M=2*L+2, k=2*L+1)[0]

    #PDOS for spin down
    plt.figure(2)
    plt.subplot(211)
    plt.plot(x,  sum([np.dot(states[:, i], orb_up)**2 * gaussian(eta, x-values[i]) for i in range(2*L)]), label = 'spin up')
    plt.legend(loc = 0)
    plt.subplot(212)
    plt.plot(x,  sum([np.dot(states[:, i], orb_down)**2 * gaussian(eta, x-values[i]) for i in range(2*L)]), label = 'spin down')
    plt.legend(loc = 0)

    #Parameters for assignment c
    e = [-t, 0, t]
    L = 20

    dm = [np.zeros((2*L+2, 2*L+2)) for q in range(3)]

    for p in range(3):
        #Computation of Eigenvalues/states of the hamiltonian
        ham = hamiltonian(t, v, e[p], L)
        values, states = la.eigh(ham)

        #Calculation of the density matrix
        for i in range(2*L+2):
            for j in range(2*L+2):
                dm[p][i, j] = sum([np.dot(np.eye(1, M=2*L+2, k=i)[0], states[:, n]) \
                                   * 0.5 * np.dot(states[:, n], np.eye(1, M=2*L+2, k=j)[0]) for n in range(2*L+2)])

    #Colorplot of the density,matrix
    plt.figure(3)
    for p in range(3):
        plt.subplot(1, 3, p+1)
        plt.imshow(dm[p], interpolation = 'none', aspect = 'auto')
        plt.colorbar()
             
    

    #show all the figures
    plt.show()


    

#Start of the main function
if __name__=='__main__':
    main()
