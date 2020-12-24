import numpy as np
from tools import tmul, hosvd, training_tensor, read_reshape

c = None
h = None


def tensor_a1(test_image, t, norm=2, tol=0.75):
    global c
    global h
    if c is None:
        s, u1, u2, h, _ = hosvd(t)
        c = tmul(tmul(s, u1, 1), u2, 2)
    else:
        pass
    while tol < 1:
        for e in range(0, 8):
            c_e = c[:, e, :].T
            a_e = np.dot(np.linalg.pinv(c_e), test_image)
            for p in range(0, 40):
                #print(np.linalg.norm(h[p] - a_e))
                if np.linalg.norm(h[p] - a_e) < tol:
                    return p + 1
        tol = tol + 0.02
    print('Persoana nu se află in baza de date!')
    return None


#test_img = read_reshape(r'C:\Users\info\Desktop\Info Anul III\ACS\Proiect\db\s37\9.pgm').astype(np.float64)
#print(tensor_a1(test_img))