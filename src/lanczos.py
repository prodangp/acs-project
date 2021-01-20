import numpy as np

from tools import center_data
from nn import nn
from time import time

cA = None
avg_image = None
E = None
Y = None
ppt = 0


def hqpb(A, k=100):
    b = []
    q = []
    w = []
    a = []
    resolution = len(A.T[0])
    b.append(0)
    q.append(np.array([[0] * resolution]).real.astype(np.float64))
    q.append(np.array([[0] * resolution]).real.astype(np.float64))
    q[1][0][0] = 1
    for i in range(0, k):
        w.append(np.dot(np.dot(q[i + 1], A), A.T) - b[i] * q[i])
        a.append(np.dot(w[i], q[i+1].T)[0][0])
        w[i] = w[i] - a[i] * q[i + 1]
        b.append(np.linalg.norm(w[i]))
        q.append(w[i] / b[i+1])
    pb = []
    for i in range(2, k+1):
        pb.append(q[i][0])
    return np.array(pb).T.real.astype(np.float64)


def test(test_image, A, norm=2, k=80):
    global cA
    global avg_image
    global E
    global Y
    global ppt
    if cA is None:
        tic = time()
        cA, avg_image = center_data(A)
        cA = cA.T
        E = hqpb(cA, k)  # High-Quality Pseudo-basis
        Y = np.dot(E.T, cA)
        toc = time()
        ppt = toc - tic
        print("pre-processing time = ", (toc - tic) * 1000, ' ms')
    else:
        pass
    c_test_image = np.array([np.array(test_image-avg_image)]).T
    test_pr = np.dot(E.T, c_test_image)
    return nn(test_pr.T, Y.T, norm)



