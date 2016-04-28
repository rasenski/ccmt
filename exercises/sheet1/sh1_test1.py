import numpy as np

def main():
    N = 100

    X = np.array([[0,1j],[-1j,0]])
    Y = np.array([[2j,0],[0,-2j]])

    M_X = np.kron(np.eye(4), X)
    M_Y = np.kron(np.eye(4), Y)
    M_Y_T = M_Y.T.conj()

    H = np.kron(np.eye(N), M_X)
    H += np.kron(np.eye(N, k=1), M_Y)
    H += np.kron(np.eye(N, k=-1), M_Y_T)
    
    print(H)

if __name__ == "__main__":
    main()
