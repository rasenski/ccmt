import numpy as np
import matplotlib.pyplot as plt

t, xhi = np.mgrid[0:2*np.pi:0.01, 0:1:0.1]

x = np.exp(-xhi*t)*np.cos(np.sqrt(1-xhi*xhi)*t)

plt.plot(t,x)
plt.show()
