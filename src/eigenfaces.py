from time import time
from tools import *
from nn import nn

cA = None
avg_image = None
E = None
Y = None
ppt = 0


def eigenvalues(A, k=60):
    n, m = A.shape
    C = np.dot(A, A.T) if m > n else np.dot(A.T, A)
    w, v = np.linalg.eig(C)  # w - eigenvalues and v - eigenvectors
    w = w.real.astype(np.float64)
    v = v.real.astype(np.float64)
    indices_sort = np.argsort(w)[-k:]
    E = []
    for ind in indices_sort:
        E.append(np.dot(A, v[ind]))
    return np.array(E).T


def eigenfaces(test_image, A, norm, k=60):
    global cA
    global avg_image
    global E
    global Y
    global ppt
    # pre processing stage
    if cA is None:
        tic = time()
        cA, avg_image = center_data(A)
        cA = cA.T
        E = eigenvalues(cA, k)  # High-Quality Pseudo-basis
        Y = np.dot(E.T, cA)
        toc = time()
        ppt = toc - tic
        print("pre-processing time = ", (toc - tic) * 1000, ' ms')
    else:
        pass
    # testing stage
    c_test_image = np.array([np.array(test_image) - avg_image]).T
    test_pr = np.dot(E.T, c_test_image)
    p = nn(test_pr.T, Y.T, norm)
    return p


def eigenfaces_reduced(test_image, A, norm=2, k=60):
    global cA
    global avg_image
    global E
    global Y
    global ppt
    # pre processing stage
    if cA is None:
        tic = time()
        cA, avg_image = center_data(reduce(A))
        cA = cA.T
        E = eigenvalues(cA, k)
        Y = np.dot(E.T, cA)
        toc = time()
        ppt = toc - tic
        print("pre-processing time = ", (toc - tic) * 1000, ' ms')
    else:
        pass
    # testing stage
    c_test_image = np.array([np.array(test_image) - avg_image]).T
    test_pr = np.dot(E.T, c_test_image)
    return nn(test_pr.T, Y.T, norm, reduced=True)
