import numpy as np

class density_of_states():

    def __init__(self, efermi, energy, eta):            ##Initialize the energy difference and the
        self.E = self.__ediff(efermi, energy)
        self.eta = eta
        self.elen = np.size(energy)                     ##Amount of sites

### __ediff produces an array where every entry holds all energy differences of the spectrum to the current fermi energy
    def __ediff(self, efermi, energy):                  ##Create a matrix with all deltas between efermi and the spectrum
        buff = []
        for e in np.nditer(efermi):
            buff.append( e - energy )
        return np.array(buff)                           ##Transform the list of matrices into a numpy array

    def gaussian(self):
        density = 1./(self.eta * np.sqrt(np.pi)) * np.exp(- (self.E * self.E)/(self.eta*self.eta))
        return np.sum(np.sum(density, axis=1), axis=1)/self.elen    ##Here we first sum over the gaussians for every k point
                                                                    ##Then we sum over all bands
    def lorentzian(self):
        density = self.eta/(np.pi * (self.E * self.E + self.eta * self.eta))
        return np.sum(np.sum(density, axis=1), axis=1)/self.elen
