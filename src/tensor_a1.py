from time import time
import numpy as np
from tools import tmul, hosvd

c = None
h = None
ppt = 0


def tensor_a1(test_image, t, norm=2, tol=0.78):
    global c
    global h
    global ppt
    if c is None:
        tic = time()
        s, u1, u2, h, _ = hosvd(t)
        c = tmul(tmul(s, u1, 1), u2, 2)
        toc = time()
        ppt = toc - tic
        print("pre-processing time = ", (toc - tic) * 1000, ' ms')
    else:
        pass
    while tol < 1:
        for e in range(0, 8):
            c_e = c[:, e, :].T
            a_e = np.dot(np.linalg.pinv(c_e), test_image)
            for p in range(0, 40):
                if norm == 'cos':
                    if 1 - np.dot(h[p], a_e) / (np.linalg.norm((h[p]) * np.linalg.norm(a_e))) < tol:
                        return p + 1
                else:
                    if np.linalg.norm(h[p] - a_e, norm) < tol:
                        return p + 1
        tol = 2

    return -1


#test_img = read_reshape(r'C:\Users\info\Desktop\Info Anul III\ACS\Proiect\db\s37\9.pgm').astype(np.float64)
#print(tensor_a1(test_img))