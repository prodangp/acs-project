import cv2
import numpy as np
path = r'C:\Users\info\Desktop\Info Anul III\ACS\Proiect'
test_img_per_person = 2
TI = 8
persons_no = 40
img_no = test_img_per_person * persons_no

def read_reshape(pgm):
    data = cv2.imread(pgm)
    data = np.reshape(data, (10304, 3))
    data = data.T
    return data[0]


def training_matrix():
    A=[]
    for i in range(1, 41):
        for j in range(1, 9):
            pgm_path = path + r'\db\s' + str(i) + "\\" + str(j) + '.pgm'
            img = read_reshape(pgm_path).astype(np.float64)
            A.append(img)

    A = np.array(A)
    return A


def find_person(k):
    return int(k/8) + 1


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
        if i == TI:
            i = 0
            for pixel in range(0, resolution):
                avg[pixel] = avg[pixel] / TI
            x.append(avg)
            avg = np.array([0.0] * resolution).astype(np.float64)
        for pixel in range(0, resolution):
            avg[pixel] += img[pixel]
        i += 1
    for pixel in range(0, resolution):
        avg[pixel] = avg[pixel] / TI
    x.append(avg)
    x = np.array(x).astype(np.float64)
    return x


def center_data(A):
    avg = avg_img(A)
    for img_no in range(0, len(A)):
        A[img_no] = A[img_no] - avg
    return A, avg




