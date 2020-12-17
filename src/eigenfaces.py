from tools import *
from nn import nn

cA = None
avg_image = None
E = None
Y = None


def eigenvalues(A, k=30):
    n, m = A.shape
    print(n, m)
    C = np.dot(A, A.T) if m > n else np.dot(A.T, A)
    w, v = np.linalg.eig(C)  # w - eigenvalues and v - eigenvectors
    w = w.real.astype(np.float64)
    v = v.real.astype(np.float64)
    indices_sort = np.argsort(w)[-k:]
    E = []
    for ind in indices_sort:
        E.append(np.dot(A, v[ind]))
    return np.array(E).T


def eigenfaces(test_image, A, norm=2):
    global cA
    global avg_image
    global E
    global Y
    if cA is None:
        cA, avg_image = center_data(A)
        cA = cA.T
        E = eigenvalues(cA)  # High-Quality Pseudo-basis
        Y = np.dot(E.T, cA)
    else:
        pass
    c_test_image = np.array([np.array(test_image) - avg_image]).T
    test_pr = np.dot(E.T, c_test_image)
    return nn(test_pr.T, Y.T, norm)


def eigenfaces_reduced(test_image, A, norm=2):
    global cA
    global avg_image
    global E
    global Y
    if cA is None:
        cA, avg_image = center_data(reduce(A))
        cA = cA.T
        E = eigenvalues(cA)
        Y = np.dot(E.T, cA)
    else:
        pass
    c_test_image = np.array([np.array(test_image) - avg_image]).T
    test_pr = np.dot(E.T, c_test_image)
    return nn(test_pr.T, Y.T, norm, reduced=True)


#test_img = read_reshape(r'C:\Users\info\Desktop\Info Anul III\ACS\Proiect\db\s29\9.pgm').astype(np.float64)
#A = training_matrix()
#print(eigenfaces(test_img, A))