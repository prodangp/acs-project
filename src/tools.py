import os
import cv2
import numpy as np

path = r'..'
test_img_per_person = 2
persons = 40
img_no = test_img_per_person * persons
resolution = 10304
TEST_IMAGES = 8

def read_reshape(img, resolution=10304):
    data = cv2.imread(img)
    data = np.reshape(data, (resolution, 3))
    data = data.T
    return data[0]


def training_matrix(path_db=r'..\db\s',persons=40, TI=8, extension='.pgm', resolution=10304):
    A=[]
    global TEST_IMAGES
    TEST_IMAGES = TI
    for i in range(1, persons + 1):
        for j in range(1, TI + 1):
            img_path = path_db + str(i) + "\\" + str(j) + extension
            img = read_reshape(img_path, resolution).astype(np.float64)
            A.append(img)
    A = np.array(A)
    return A


def training_tensor(path_db=r'..\db\s',persons=40, TI=8, extension='.pgm'):
    t = []
    global TEST_IMAGES
    TEST_IMAGES = TI
    for i in range(1, persons + 1):
        p = []
        for j in range(1, TI + 1):
            pgm_path = path_db + str(i) + "\\" + str(j) + extension
            img = read_reshape(pgm_path).astype(np.float64)
            p.append(img)
        p = np.array(p)
        t.append(p)
    t = np.array(t)
    return t


def unfold(t, mode):
    if mode == 1:
        t1 = t[:, 0, :].T
        for i in range(1, TEST_IMAGES):
            t1 = np.concatenate((t1, t[:, i, :].T), axis=1)
        return t1
    if mode == 2:
        t2 = t[0, :, :]
        for j in range(1, persons):
            t2 = np.concatenate((t2, t[j, :, :]), axis=1)
        return t2
    if mode == 3:
        t3 = []
        for j in range(0, persons):
            p = []
            for k in range(0, TEST_IMAGES):
               p = np.concatenate((p, t[j][k]))
            t3.append(p)
        return np.array(t3)


def fold(u, mode):
    l = resolution
    m = TEST_IMAGES
    n = persons
    t = [None] * n
    for i in range(0,n):
        t[i] = []
    if mode == 1:
        u = u.T
        e = 0
        p = 0
        while e < m:
            t[p].append(u[p + e * n])
            p += 1
            if p == n:
                p = 0
                e += 1
        return np.array(t)
    if mode == 2:
        e = 0
        p = 0
        while e < m:
            t[p].append(u[e][p * l:(p + 1) * l])
            p += 1
            if p == n:
                p = 0
                e += 1
        return np.array(t)
    if mode == 3:
        e = 0
        p = 0
        while p < n:
            t[p].append(u[p][e * l:(e + 1) * l])
            e += 1
            if e == m:
                e = 0
                p += 1
        return np.array(t)


def hosvd(t):
    u1, s1, v1 = np.linalg.svd(unfold(t, 1))
    u2, s2, v2 = np.linalg.svd(unfold(t, 2), full_matrices=False)
    u3, s3, v3 = np.linalg.svd(unfold(t, 3), full_matrices=False)
    s = tmul(tmul(tmul(t, u1.T, 1), u2.T, 2), u3.T, 3)
    tt = tmul(tmul(tmul(s, u1, 1), u2, 2), u3, 3)
    return s, u1, u2, u3, tt


def tmul(t, m, i):
    return fold(np.dot(m, unfold(t, i)), i)


def find_person(k):
    return int(k/TEST_IMAGES) + 1


def avg_img(A):
    avg = [0.0] * len(A[0])
    avg = np.array(avg).astype(np.float64)
    for img in A:
        for pixel in range(0, len(img)):
            avg[pixel] += img[pixel]
    for pixel in range(0, len(avg)):
        avg[pixel] = avg[pixel] / len(A)
    return avg


def reduce(A):
    x = []
    i = 0
    resolution = len(A[0])
    avg = [0.0] * resolution
    avg = np.array(avg).astype(np.float64)
    for img in A:
        if i == TEST_IMAGES:
            i = 0
            for pixel in range(0, resolution):
                avg[pixel] = avg[pixel] / TEST_IMAGES
            x.append(avg)
            avg = np.array([0.0] * resolution).astype(np.float64)
        for pixel in range(0, resolution):
            avg[pixel] += img[pixel]
        i += 1
    for pixel in range(0, resolution):
        avg[pixel] = avg[pixel] / TEST_IMAGES
    x.append(avg)
    x = np.array(x).astype(np.float64)
    return x


def center_data(A):
    avg = avg_img(A)
    for img_no in range(0, len(A)):
        A[img_no] = A[img_no] - avg
    return A, avg



# def tmul_by_hard(t, m, i):
#     r1 = []
#     if i == 1:
#         for i in range(0, 40):
#             r2 = []
#             for j in range(0, 8):
#                 r2.append(np.dot(m.T, t[i][j]))
#             r1.append(r2)
#         return np.array(r1)
#     if i == 2:
#         for i in range(0, 40):
#             r2 = []
#             for j in range(0, 8):
#                 r2.append(np.dot(t[i].T, m.T[j]))
#             r1.append(r2)
#         return np.array(r1)
#     if i == 3:
#         for i in range(0, 40):
#             r2 = []
#             for j in range(0, 8):
#                 r3 = []
#                 for k in range(0, 10304):
#                     s = 0
#                     for l in range(0, 40):
#                         s += m[l][i]*t[l][j][k]
#                     r3.append(s)
#                 r2.append(r2)
#             r1.append(r2)
#         return np.array(r1)

